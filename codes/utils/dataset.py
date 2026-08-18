import os
import copy
from collections.abc import Mapping
import json
import random
import pickle
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from urllib.parse import unquote
from codes.utils.functions import get_project_root, resolve_project_path
from codes.utils.triple_builder import build_dataset_entity_triples
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def _load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = json.load(file)
            data.extend(file_data)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as file:
                file_data = json.load(file)
                data.extend(file_data)
    return data

def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)

def _stack_tokenized_dicts(input_dict_list):
    if not input_dict_list:
        return {}
    batch = {}
    for key in input_dict_list[0].keys():
        values = [sample[key] for sample in input_dict_list]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.tensor(values, dtype=torch.long)
    return batch

def _unwrap_cached_dataset(cache_obj):
    if isinstance(cache_obj, dict) and 'data' in cache_obj:
        return cache_obj['data']
    return cache_obj

def _get_cached_text_length(sample):
    if not isinstance(sample, Mapping) or 'input_ids' not in sample:
        return None
    input_ids = sample['input_ids']
    if torch.is_tensor(input_ids):
        return int(input_ids.shape[0])
    if hasattr(input_ids, '__len__'):
        return len(input_ids)
    return None

def _check_cached_dataset_compatibility(cache_obj, expected_text_length):
    cached_samples = _unwrap_cached_dataset(cache_obj)
    if not isinstance(cached_samples, list) or not cached_samples:
        return (False, cached_samples, None)
    cached_text_length = _get_cached_text_length(cached_samples[0])
    if cached_text_length is None:
        return (True, cached_samples, None)
    is_compatible = cached_text_length == expected_text_length
    return (is_compatible, cached_samples, cached_text_length)

class DataModuleForSKAMEL(pl.LightningDataModule):

    def __init__(self, args):
        super(DataModuleForSKAMEL, self).__init__()
        self.args = args
        self.project_root = getattr(self.args, 'project_root', get_project_root())
        self.pretrained_model_path = resolve_project_path(self.project_root, self.args.pretrained_model)
        self.kb_img_folder = resolve_project_path(self.project_root, self.args.data.kb_img_folder)
        self.mention_img_folder = resolve_project_path(self.project_root, self.args.data.mention_img_folder)
        self.qid2id_path = resolve_project_path(self.project_root, self.args.data.qid2id)
        self.entity_path = resolve_project_path(self.project_root, self.args.data.entity)
        self.train_file_path = resolve_project_path(self.project_root, self.args.data.train_file)
        self.dev_file_path = resolve_project_path(self.project_root, self.args.data.dev_file)
        self.test_file_path = resolve_project_path(self.project_root, self.args.data.test_file)
        self.tokenizer = CLIPProcessor.from_pretrained(self.pretrained_model_path).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(self.pretrained_model_path).feature_extractor
        with open(self.qid2id_path, 'r', encoding='utf-8') as file:
            self.qid2id = json.loads(file.readline())
        self.raw_kb_entity = sorted(_load_json_file(self.entity_path), key=lambda sample: sample['id'])
        self.use_triple = bool(getattr(self.args.model, 'use_triple', False))
        self.entity_triple_records, self.entity_triple_token_cache = self._load_entity_triple_artifacts()
        self.kb_entity = self.setup_dataset_for_entity(self.entity_path, self.raw_kb_entity)
        self.kb_id2entity = {raw_entity['id']: entity for raw_entity, entity in zip(self.raw_kb_entity, self.kb_entity)}
        self.train_data = self.setup_dataset_for_mention(self.train_file_path, _load_json_file(self.train_file_path))
        self.val_data = self.setup_dataset_for_mention(self.dev_file_path, _load_json_file(self.dev_file_path))
        self.test_data = self.setup_dataset_for_mention(self.test_file_path, _load_json_file(self.test_file_path))

    def _triple_shape(self):
        triple_max_num = int(getattr(self.args.data, 'triple_max_num', 3))
        triple_text_max_length = int(getattr(self.args.data, 'triple_text_max_length', 16))
        return (triple_max_num, triple_text_max_length)

    def _default_triple_artifact_paths(self):
        entity_dir = os.path.dirname(self.entity_path)
        return (os.path.join(entity_dir, 'kb_entity_triples.json'), os.path.join(entity_dir, 'kb_entity_triple_token_cache.pkl'))

    def _resolve_triple_artifact_paths(self):
        triple_file_cfg = getattr(self.args.data, 'entity_triple_file', None)
        triple_cache_cfg = getattr(self.args.data, 'entity_triple_token_cache', None)
        default_triple_file, default_triple_cache = self._default_triple_artifact_paths()
        triple_file_path = resolve_project_path(self.project_root, triple_file_cfg) if triple_file_cfg else default_triple_file
        triple_cache_path = resolve_project_path(self.project_root, triple_cache_cfg) if triple_cache_cfg else default_triple_cache
        return (triple_file_path, triple_cache_path)

    def _get_triple_build_meta(self):
        triple_max_num, triple_text_max_length = self._triple_shape()
        return {'rule_id': 'skamel', 'triple_max_num': int(triple_max_num), 'triple_text_max_length': int(triple_text_max_length)}

    def _is_triple_cache_compatible(self, triple_cache, expected_meta):
        if not isinstance(triple_cache, dict):
            return False
        if triple_cache.get('meta') != expected_meta:
            return False
        required_keys = {'input_ids', 'attention_mask', 'triple_valid_mask'}
        if not required_keys.issubset(set(triple_cache.keys())):
            return False
        input_ids = np.asarray(triple_cache['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(triple_cache['attention_mask'], dtype=np.int64)
        triple_valid_mask = np.asarray(triple_cache['triple_valid_mask'], dtype=np.int64)
        num_entity = len(self.raw_kb_entity)
        triple_max_num = expected_meta['triple_max_num']
        triple_text_max_length = expected_meta['triple_text_max_length']
        if input_ids.ndim != 3 or attention_mask.ndim != 3 or triple_valid_mask.ndim != 2:
            return False
        if input_ids.shape != attention_mask.shape:
            return False
        if input_ids.shape[0] != num_entity or triple_valid_mask.shape[0] != num_entity:
            return False
        if input_ids.shape[1] != triple_max_num or triple_valid_mask.shape[1] != triple_max_num:
            return False
        if input_ids.shape[2] != triple_text_max_length:
            return False
        return True

    def _empty_triple_fields(self):
        triple_max_num, triple_text_max_length = self._triple_shape()
        return {'triple_input_ids': np.zeros((triple_max_num, triple_text_max_length), dtype=np.int64), 'triple_attention_mask': np.zeros((triple_max_num, triple_text_max_length), dtype=np.int64), 'triple_valid_mask': np.zeros((triple_max_num,), dtype=np.int64)}

    def _load_entity_triple_artifacts(self):
        if not self.use_triple:
            return (None, None)
        triple_file_path, triple_cache_path = self._resolve_triple_artifact_paths()
        expected_meta = self._get_triple_build_meta()
        force_rebuild = _as_bool(getattr(self.args.data, 'force_rebuild_entity_triples', False), default=False)
        triple_cache = None
        triple_records = None

        def _build_triples():
            print(f'[DataModule] building entity triples:\n  entity={self.entity_path}\n  triple_file={triple_file_path}\n  triple_cache={triple_cache_path}')
            build_dataset_entity_triples(kb_entity_path=self.entity_path, output_json_path=triple_file_path, output_token_cache_path=triple_cache_path, tokenizer=self.tokenizer, triple_max_num=expected_meta['triple_max_num'], triple_text_max_length=expected_meta['triple_text_max_length'])
        need_rebuild = force_rebuild
        if force_rebuild:
            print('[DataModule] force_rebuild_entity_triples=true, rebuilding triple artifacts.')
        if not os.path.exists(triple_file_path) or not os.path.exists(triple_cache_path):
            need_rebuild = True
        if not need_rebuild:
            try:
                with open(triple_cache_path, 'rb') as file:
                    triple_cache = pickle.load(file)
            except Exception as exc:
                print(f'[DataModule] failed to load triple cache: {triple_cache_path}, err={exc}')
                need_rebuild = True
        if not need_rebuild and (not self._is_triple_cache_compatible(triple_cache, expected_meta)):
            print(f'[DataModule] triple cache incompatible with current config, rebuilding: {triple_cache_path}')
            need_rebuild = True
        if not need_rebuild:
            try:
                with open(triple_file_path, 'r', encoding='utf-8') as file:
                    triple_records = json.load(file)
                if len(triple_records) != len(self.raw_kb_entity):
                    print(f'[DataModule] triple record count mismatch, rebuilding: {triple_file_path}')
                    need_rebuild = True
            except Exception as exc:
                print(f'[DataModule] failed to load triple records: {triple_file_path}, err={exc}')
                need_rebuild = True
        if need_rebuild:
            _build_triples()
            triple_cache = None
            triple_records = None
        if triple_records is None:
            with open(triple_file_path, 'r', encoding='utf-8') as file:
                triple_records = json.load(file)
        if triple_cache is None:
            with open(triple_cache_path, 'rb') as file:
                triple_cache = pickle.load(file)
        print(f'[DataModule] loaded triple artifacts from {triple_file_path} and {triple_cache_path}')
        return (triple_records, {'input_ids': np.asarray(triple_cache['input_ids'], dtype=np.int64), 'attention_mask': np.asarray(triple_cache['attention_mask'], dtype=np.int64), 'triple_valid_mask': np.asarray(triple_cache['triple_valid_mask'], dtype=np.int64)})

    def _ensure_entity_triple_fields(self, input_data):
        if not self.use_triple:
            return input_data
        empty_fields = self._empty_triple_fields()
        has_cache = self.entity_triple_token_cache is not None
        for row_idx, sample in enumerate(input_data):
            if has_cache:
                sample['triple_input_ids'] = self.entity_triple_token_cache['input_ids'][row_idx]
                sample['triple_attention_mask'] = self.entity_triple_token_cache['attention_mask'][row_idx]
                sample['triple_valid_mask'] = self.entity_triple_token_cache['triple_valid_mask'][row_idx]
            else:
                sample['triple_input_ids'] = empty_fields['triple_input_ids']
                sample['triple_attention_mask'] = empty_fields['triple_attention_mask']
                sample['triple_valid_mask'] = empty_fields['triple_valid_mask']
        return input_data

    def setup_dataset_for_entity(self, path, data):
        pkl_path = path[0:path.rfind('.')] + '.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                cache_obj = pickle.load(file)
            is_compatible, input_data, cached_text_length = _check_cached_dataset_compatibility(cache_obj, self.args.data.text_max_length)
            if is_compatible:
                return self._ensure_entity_triple_fields(input_data)
            print(f'[DataModule] entity cache length mismatch; rebuilding: {pkl_path} (cached={cached_text_length}, expected={self.args.data.text_max_length})')
        input_data = []
        for row_idx, sample_dict in enumerate(tqdm(data, desc='PreProcessing')):
            sample_type = sample_dict['type']
            if sample_type == 'entity':
                entity = unquote(sample_dict.pop('entity_name'))
                desc = sample_dict.pop('desc')
                input_text = entity + ' [SEP] ' + desc
                input_dict = self.tokenizer(input_text, padding='max_length', max_length=self.args.data.text_max_length, truncation=True)
            else:
                input_dict = {}
            input_dict['img_list'] = sample_dict['image_list']
            if self.use_triple:
                if self.entity_triple_token_cache is not None:
                    input_dict['triple_input_ids'] = self.entity_triple_token_cache['input_ids'][row_idx]
                    input_dict['triple_attention_mask'] = self.entity_triple_token_cache['attention_mask'][row_idx]
                    input_dict['triple_valid_mask'] = self.entity_triple_token_cache['triple_valid_mask'][row_idx]
                else:
                    input_dict.update(self._empty_triple_fields())
            input_dict['sample_type'] = 0 if sample_type == 'entity' else 1
            if 'answer' in sample_dict:
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            input_data.append(input_dict)
        with open(pkl_path, 'wb') as file:
            pickle.dump(input_data, file)
        return input_data

    def setup_dataset_for_mention(self, path, data):
        pkl_path = path[0:path.rfind('.')] + '.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                cache_obj = pickle.load(file)
            is_compatible, input_data, cached_text_length = _check_cached_dataset_compatibility(cache_obj, self.args.data.text_max_length)
            if is_compatible:
                return input_data
            print(f'[DataModule] mention cache length mismatch; rebuilding: {pkl_path} (cached={cached_text_length}, expected={self.args.data.text_max_length})')
        input_data = []
        for sample_dict in tqdm(data, desc='PreProcessing'):
            sample_type = 1
            _ = unquote(sample_dict.pop('entities'))
            mention = unquote(sample_dict.pop('mentions'))
            text = sample_dict.pop('sentence')
            desc = sample_dict.pop('desc')
            input_text = mention + ' [SEP] ' + text + ' [SEP] ' + desc
            input_dict = self.tokenizer(input_text, padding='max_length', max_length=self.args.data.text_max_length, truncation=True)
            input_dict['img_list'] = [sample_dict['imgPath']] if sample_dict['imgPath'] != '' else []
            input_dict['sample_type'] = sample_type
            if 'answer' in sample_dict:
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            if sample_dict['answer'] == 'nil':
                continue
            input_data.append(input_dict)
        with open(pkl_path, 'wb') as file:
            pickle.dump(input_data, file)
        return input_data

    def choose_image(self, sample_type, img_list, is_eval=False):
        if len(img_list):
            img_name = random.choice(img_list)
            if is_eval:
                img_name = img_list[0]
            if sample_type == 1:
                img_name = img_name.split('/')[-1].split('.')[0] + '.jpg'
            try:
                img_root = self.kb_img_folder if sample_type == 0 else self.mention_img_folder
                img_path = os.path.join(img_root, img_name)
                image = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze()
            except Exception:
                pixel_values = torch.zeros((3, 224, 224))
        else:
            pixel_values = torch.zeros((3, 224, 224))
        return pixel_values

    def train_collator(self, samples):
        img_list, sample_type, input_dict_list = ([], [], [])
        pixel_values, gt_ent_id = ([], [])
        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))
            input_dict_list.append(sample)
        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx]))
        input_dict = _stack_tokenized_dicts(input_dict_list)
        input_dict['pixel_values'] = torch.stack(pixel_values)
        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in gt_ent_id]
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = ([], [], [], [])
        ent_triple_input_ids, ent_triple_attention_mask, ent_triple_valid_mask = ([], [], [])
        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            if self.use_triple:
                empty_fields = self._empty_triple_fields()
                ent_triple_input_ids.append(ent_dict.pop('triple_input_ids', empty_fields['triple_input_ids']))
                ent_triple_attention_mask.append(ent_dict.pop('triple_attention_mask', empty_fields['triple_attention_mask']))
                ent_triple_valid_mask.append(ent_dict.pop('triple_valid_mask', empty_fields['triple_valid_mask']))
            ent_input_dict_list.append(ent_dict)
        for idx, _ in enumerate(ent_input_dict_list):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))
        ent_empty_img_flag = torch.tensor([not len(entity_img_list) for entity_img_list in ent_img_list], dtype=torch.bool)
        ent_input_dict = _stack_tokenized_dicts(ent_input_dict_list)
        ent_input_dict['pixel_values'] = torch.stack(ent_pixel_values)
        ent_input_dict['empty_img_flag'] = ent_empty_img_flag
        if self.use_triple:
            ent_input_dict['triple_input_ids'] = torch.tensor(np.stack(ent_triple_input_ids), dtype=torch.long)
            ent_input_dict['triple_attention_mask'] = torch.tensor(np.stack(ent_triple_attention_mask), dtype=torch.long)
            ent_input_dict['triple_valid_mask'] = torch.tensor(np.stack(ent_triple_valid_mask), dtype=torch.long)
        for key, value in ent_input_dict.items():
            input_dict[f'ent_{key}'] = value
        return input_dict

    def eval_collator(self, samples):
        img_list, sample_type, input_dict_list = ([], [], [])
        pixel_values, gt_ent_id = ([], [])
        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))
            input_dict_list.append(sample)
        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))
        input_dict = _stack_tokenized_dicts(input_dict_list)
        input_dict['pixel_values'] = torch.stack(pixel_values)
        input_dict['answer'] = torch.tensor(gt_ent_id, dtype=torch.long)
        return input_dict

    def entity_collator(self, samples):
        pixel_values, img_list, sample_type, input_dict_list = ([], [], [], [])
        triple_input_ids, triple_attention_mask, triple_valid_mask = ([], [], [])
        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            if self.use_triple:
                empty_fields = self._empty_triple_fields()
                triple_input_ids.append(sample.pop('triple_input_ids', empty_fields['triple_input_ids']))
                triple_attention_mask.append(sample.pop('triple_attention_mask', empty_fields['triple_attention_mask']))
                triple_valid_mask.append(sample.pop('triple_valid_mask', empty_fields['triple_valid_mask']))
            input_dict_list.append(sample)
        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))
        input_dict = _stack_tokenized_dicts(input_dict_list)
        input_dict['pixel_values'] = torch.stack(pixel_values)
        if self.use_triple:
            input_dict['triple_input_ids'] = torch.tensor(np.stack(triple_input_ids), dtype=torch.long)
            input_dict['triple_attention_mask'] = torch.tensor(np.stack(triple_attention_mask), dtype=torch.long)
            input_dict['triple_valid_mask'] = torch.tensor(np.stack(triple_valid_mask), dtype=torch.long)
        return input_dict

    def entity_dataloader(self):
        return DataLoader(self.kb_entity, batch_size=self.args.data.embed_update_batch_size, num_workers=self.args.data.num_workers, shuffle=False, collate_fn=self.entity_collator)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.data.batch_size, num_workers=self.args.data.num_workers, shuffle=True, collate_fn=self.train_collator)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.data.eval_batch_size, num_workers=self.args.data.num_workers, shuffle=False, collate_fn=self.eval_collator)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.data.eval_batch_size, num_workers=self.args.data.num_workers, shuffle=False, collate_fn=self.eval_collator)
