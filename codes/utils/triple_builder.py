import json
import os
import pickle
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
STRUCTURAL_RULE_ID = 'skamel'
GENDER_RE = re.compile('\\bGender\\s+(male|female|intersex)\\b', re.I)
OCCUPATION_RE = re.compile('\\bOccupation\\s+([^\\.]+)', re.I)
DESC_LOC_IN_RE = re.compile('^(city|town|village|commune|municipality|county|railway station|river|human settlement|district|province|department|island|mountain|lake|airport|port|university|school|hospital|bridge|stadium|park|road|highway|motorway|station|canton|prefecture) in (.+)$', re.I)
DESC_LOC_OF_RE = re.compile('^(municipality|district|province|department|county|city|town|village) of (.+)$', re.I)
DESC_CAPITAL_OF_RE = re.compile('^(capital(?: and largest city)?|largest city) of (.+)$', re.I)
DESC_WORK_BY_RE = re.compile('^(album|song|novel|book|film|television series|tv series|poem|painting|opera|single|video game) by (.+)$', re.I)
DESC_DIRECTED_BY_RE = re.compile('^(film|movie|television series|tv series) directed by (.+)$', re.I)
DESC_EXTRA_RULES: Tuple[Tuple[str, re.Pattern, str, int], ...] = (('located in', DESC_LOC_IN_RE, 'desc_loc_in', 12), ('part of', DESC_LOC_OF_RE, 'desc_loc_of', 12), ('capital of', DESC_CAPITAL_OF_RE, 'desc_capital_of', 10), ('creator', DESC_WORK_BY_RE, 'desc_work_by', 8), ('director', DESC_DIRECTED_BY_RE, 'desc_directed_by', 6))
DESC_DEFINITION_RULES: Tuple[Tuple[str, re.Pattern, str, int], ...] = (('instance of', re.compile('^(?:a |an )?type of (.+)$', re.I), 'desc_type_of', 10), ('instance of', re.compile('^(?:a |an )?form of (.+)$', re.I), 'desc_form_of', 10), ('instance of', re.compile('^(?:a |an )?series of (.+)$', re.I), 'desc_series_of', 10), ('instance of', re.compile('^(?:a |an )?class of (.+)$', re.I), 'desc_class_of', 10), ('instance of', re.compile('^(?:a |an )?group of (.+)$', re.I), 'desc_group_of', 10), ('instance of', re.compile('^(?:a |an )?set of (.+)$', re.I), 'desc_set_of', 10), ('instance of', re.compile('^(?:a |an )?method of (.+)$', re.I), 'desc_method_of', 10), ('instance of', re.compile('^(?:a |an )?process of (.+)$', re.I), 'desc_process_of', 10), ('part of', re.compile('^(?:a |an )?part of (.+)$', re.I), 'desc_part_of', 10), ('instance of', re.compile('^(?:a |an )?list of (.+)$', re.I), 'desc_list_of', 10))

def normalize_text(text: str) -> str:
    if text is None:
        return ''
    text = str(text).strip()
    text = re.sub('\\s+', ' ', text)
    return text.strip(' ,.;:')

def _get_desc_clause(desc_text: str, take_first_sentence: bool=True) -> str:
    desc_main = normalize_text(desc_text)
    if not desc_main:
        return ''
    if not take_first_sentence:
        return desc_main
    return normalize_text(desc_main.split('.')[0])

def _is_tail_valid(tail: str, max_tokens: int, strict_filter: bool, max_chars: Optional[int]=None) -> bool:
    tail = normalize_text(tail)
    if not tail:
        return False
    if max_chars is not None and len(tail) > max_chars:
        return False
    token_num = len(tail.split())
    if token_num < 1:
        return False
    if strict_filter and token_num > max_tokens:
        return False
    if strict_filter and (';' in tail or ':' in tail):
        return False
    return True

def _extract_secondary_instance_tail(instance_raw: str) -> str:
    parts = [normalize_text(part) for part in instance_raw.split(',') if normalize_text(part)]
    if len(parts) < 2:
        return ''
    secondary_tail = parts[1]
    secondary_tail_low = secondary_tail.lower()
    token_num = len(secondary_tail_low.split())
    if token_num > 8 or len(secondary_tail) > 64:
        return ''
    if ';' in secondary_tail or ':' in secondary_tail:
        return ''
    if secondary_tail_low.startswith('and ') or secondary_tail_low.startswith('or '):
        return ''
    if secondary_tail_low in {'or translation', 'and ireland'}:
        return ''
    return secondary_tail

def _iter_desc_extra_candidates(desc_clause: str, strict_filter: bool) -> List[Tuple[str, str, str]]:
    candidates: List[Tuple[str, str, str]] = []
    for relation, pattern, source, max_tokens in DESC_EXTRA_RULES:
        match = pattern.match(desc_clause)
        if not match:
            continue
        tail = normalize_text(match.group(2))
        if _is_tail_valid(tail, max_tokens=max_tokens, strict_filter=strict_filter):
            candidates.append((relation, tail, source))
    return candidates

def _extract_desc_definition_fallback(desc_clause: str) -> Optional[Tuple[str, str, str]]:
    for relation, pattern, source, max_tokens in DESC_DEFINITION_RULES:
        match = pattern.match(desc_clause)
        if not match:
            continue
        tail = normalize_text(match.group(1))
        if _is_tail_valid(tail, max_tokens=max_tokens, strict_filter=True, max_chars=80):
            return (relation, tail, source)
    return None

def extract_entity_triples(sample_dict: dict) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    def add_triple(relation: str, tail: str, source: str) -> None:
        tail = normalize_text(tail)
        if not tail:
            return
        pair_key = (relation.lower(), tail.lower())
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        triples.append({'relation': relation, 'tail': tail, 'source': source})
    instance_raw = normalize_text(sample_dict.get('instance', ''))
    instance_main = normalize_text(instance_raw.split(',')[0])
    if instance_main:
        add_triple('instance of', instance_main, source='instance')
    attr_text = sample_dict.get('attr', '') or ''
    gender_match = GENDER_RE.search(attr_text)
    if gender_match:
        gender = normalize_text(gender_match.group(1)).lower()
        if gender in {'male', 'female', 'intersex'}:
            add_triple('sex or gender', gender, source='attr_gender')
    occupation_match = OCCUPATION_RE.search(attr_text)
    if occupation_match:
        occupation_phrase = normalize_text(occupation_match.group(1))
        if occupation_phrase:
            add_triple('occupation', occupation_phrase, source='attr_occupation')
    secondary_instance_tail = _extract_secondary_instance_tail(instance_raw)
    if secondary_instance_tail:
        add_triple('instance of', secondary_instance_tail, source='instance_secondary')
    desc_clause = _get_desc_clause(sample_dict.get('desc', ''), take_first_sentence=True)
    if desc_clause:
        for relation, tail, source in _iter_desc_extra_candidates(desc_clause, strict_filter=True):
            add_triple(relation, tail, source=source)
    if not triples and desc_clause:
        definition_triple = _extract_desc_definition_fallback(desc_clause)
        if definition_triple:
            relation, tail, source = definition_triple
            add_triple(relation, tail, source=source)
    return triples

def build_dataset_entity_triples(kb_entity_path: str, output_json_path: str, output_token_cache_path: str, tokenizer, triple_max_num: int=3, triple_text_max_length: int=16) -> None:
    with open(kb_entity_path, 'r', encoding='utf-8') as file:
        kb_entities = json.load(file)
    kb_entities = sorted(kb_entities, key=lambda sample: sample['id'])
    num_entity = len(kb_entities)
    triple_input_ids = np.zeros((num_entity, triple_max_num, triple_text_max_length), dtype=np.int64)
    triple_attention_mask = np.zeros((num_entity, triple_max_num, triple_text_max_length), dtype=np.int64)
    triple_valid_mask = np.zeros((num_entity, triple_max_num), dtype=np.int64)
    cache_meta = {'rule_id': STRUCTURAL_RULE_ID, 'triple_max_num': int(triple_max_num), 'triple_text_max_length': int(triple_text_max_length)}
    records: List[Dict] = []
    for row_idx, sample_dict in enumerate(kb_entities):
        triples = extract_entity_triples(sample_dict)
        triples = triples[:triple_max_num]
        relations = [item['relation'] for item in triples]
        tails = [item['tail'] for item in triples]
        sources = [item['source'] for item in triples]
        triple_texts = [f'{relation} [SEP] {tail}' for relation, tail in zip(relations, tails)]
        records.append({'qid': sample_dict.get('qid', ''), 'id': sample_dict.get('id'), 'triple_relations': relations, 'triple_tails': tails, 'triple_texts': triple_texts, 'triple_sources': sources})
        for triple_idx, triple_text in enumerate(triple_texts):
            tokenized = tokenizer(triple_text, padding='max_length', max_length=triple_text_max_length, truncation=True)
            triple_input_ids[row_idx, triple_idx] = np.asarray(tokenized['input_ids'], dtype=np.int64)
            triple_attention_mask[row_idx, triple_idx] = np.asarray(tokenized['attention_mask'], dtype=np.int64)
            triple_valid_mask[row_idx, triple_idx] = 1
    output_json_dir = os.path.dirname(output_json_path)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    output_cache_dir = os.path.dirname(output_token_cache_path)
    if output_cache_dir:
        os.makedirs(output_cache_dir, exist_ok=True)
    with open(output_token_cache_path, 'wb') as file:
        pickle.dump({'input_ids': triple_input_ids, 'attention_mask': triple_attention_mask, 'triple_valid_mask': triple_valid_mask, 'meta': cache_meta}, file, protocol=pickle.HIGHEST_PROTOCOL)
