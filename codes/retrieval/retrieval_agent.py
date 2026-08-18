import math
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
from codes.backbone.outputs import PerceptionOutput
from codes.retrieval.outputs import EntityCache, RetrievalOutput

def move_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}

class RetrievalAgent:

    def __init__(self, perception_module, matching_module, num_entity: int):
        self.perception_module = perception_module
        self.matching_module = matching_module
        self.num_entity = int(num_entity)
        self.entity_cache: Optional[EntityCache] = None

    def set_num_entity(self, num_entity: int) -> None:
        self.num_entity = int(num_entity)

    def set_entity_cache(self, entity_cache: Optional[EntityCache]) -> None:
        self.entity_cache = entity_cache

    def clear_entity_cache(self) -> None:
        self.entity_cache = None

    def build_entity_cache(self, entity_dataloader, device) -> EntityCache:
        text_embeds: List[torch.Tensor] = []
        image_embeds: List[torch.Tensor] = []
        text_tokens: List[torch.Tensor] = []
        image_tokens: List[torch.Tensor] = []
        triple_cls_list: List[torch.Tensor] = []
        triple_tokens_list: List[torch.Tensor] = []
        triple_valid_mask_list: List[torch.Tensor] = []
        supports_triple = bool(getattr(self.matching_module, 'use_triple', False)) and hasattr(self.matching_module, 'encode_entity_triples')
        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='Encoding entity cache', total=len(entity_dataloader)):
                batch = move_batch_to_device(batch, device)
                triple_input_ids = batch.pop('triple_input_ids', None)
                triple_attention_mask = batch.pop('triple_attention_mask', None)
                triple_valid_mask = batch.pop('triple_valid_mask', None)
                entity_features = self.perception_module(**batch)
                if supports_triple:
                    entity_triple_cls, entity_triple_tokens, entity_triple_valid_mask = self.matching_module.encode_entity_triples(clip_text_model=self.perception_module.clip.text_model, triple_input_ids=triple_input_ids, triple_attention_mask=triple_attention_mask, triple_valid_mask=triple_valid_mask)
                    if entity_triple_cls is not None and entity_triple_tokens is not None:
                        entity_features = entity_features.with_triple(triple_cls=entity_triple_cls, triple_tokens=entity_triple_tokens, triple_valid_mask=entity_triple_valid_mask)
                text_embeds.append(entity_features.text_embeds.cpu())
                image_embeds.append(entity_features.image_embeds.cpu())
                text_tokens.append(entity_features.text_seq_tokens.cpu())
                image_tokens.append(entity_features.image_patch_tokens.cpu())
                if entity_features.triple_cls is not None and entity_features.triple_tokens is not None:
                    triple_cls_list.append(entity_features.triple_cls.cpu())
                    triple_tokens_list.append(entity_features.triple_tokens.cpu())
                    if entity_features.triple_valid_mask is not None:
                        triple_valid_mask_list.append(entity_features.triple_valid_mask.cpu())
        entity_triple_cls = torch.concat(triple_cls_list, dim=0) if triple_cls_list else None
        entity_triple_tokens = torch.concat(triple_tokens_list, dim=0) if triple_tokens_list else None
        entity_triple_valid_mask = torch.concat(triple_valid_mask_list, dim=0) if triple_valid_mask_list else None
        entity_cache = EntityCache(entity_text_embeds=torch.concat(text_embeds, dim=0), entity_image_embeds=torch.concat(image_embeds, dim=0), entity_text_seq_tokens=torch.concat(text_tokens, dim=0), entity_image_patch_tokens=torch.concat(image_tokens, dim=0), entity_triple_cls=entity_triple_cls, entity_triple_tokens=entity_triple_tokens, entity_triple_valid_mask=entity_triple_valid_mask)
        self.entity_cache = entity_cache
        self.num_entity = int(entity_cache.entity_text_embeds.size(0))
        return entity_cache

    def encode_mentions(self, batch: Dict[str, torch.Tensor], device) -> PerceptionOutput:
        mention_features = self.perception_module(**move_batch_to_device(batch, device))
        if bool(getattr(self.matching_module, 'use_triple', False)) and hasattr(self.matching_module, 'build_mention_triples'):
            mention_features = self.matching_module.build_mention_triples(mention_features)
        return mention_features

    def retrieve(self, batch: Dict[str, torch.Tensor], device, chunk_size: int, answer: Optional[torch.Tensor]=None, entity_cache: Optional[EntityCache]=None) -> RetrievalOutput:
        mention_features = self.encode_mentions(batch, device)
        return self.retrieve_from_features(mention_features=mention_features, device=device, chunk_size=chunk_size, answer=answer, entity_cache=entity_cache)

    def retrieve_from_features(self, mention_features: PerceptionOutput, device, chunk_size: int, answer: Optional[torch.Tensor]=None, entity_cache: Optional[EntityCache]=None) -> RetrievalOutput:
        cache = entity_cache or self.entity_cache
        if cache is None:
            raise RuntimeError('Entity cache is empty. Call build_entity_cache first.')
        resolved_chunk_size = max(1, int(chunk_size))
        scores = []
        text_scores = []
        image_scores = []
        cross_scores = []
        triple_scores = []
        struct_scores = []
        fusion_weights_4m = None
        for idx in range(math.ceil(self.num_entity / resolved_chunk_size)):
            start_pos = idx * resolved_chunk_size
            end_pos = min((idx + 1) * resolved_chunk_size, self.num_entity)
            chunk_entity_features = cache.slice(start_pos, end_pos, device=device)
            matching_output = self.matching_module(chunk_entity_features, mention_features)
            scores.append(matching_output.overall_score)
            text_scores.append(matching_output.text_score)
            image_scores.append(matching_output.image_score)
            cross_scores.append(matching_output.cross_score)
            if matching_output.triple_score is not None:
                triple_scores.append(matching_output.triple_score)
            elif matching_output.struct_score is not None:
                triple_scores.append(matching_output.struct_score)
            if matching_output.struct_score is not None:
                struct_scores.append(matching_output.struct_score)
            elif matching_output.triple_score is not None:
                struct_scores.append(matching_output.triple_score)
            if fusion_weights_4m is None:
                if matching_output.fusion_weights_4m is not None:
                    fusion_weights_4m = matching_output.fusion_weights_4m
                elif matching_output.fusion_weights is not None and matching_output.fusion_weights.dim() == 2 and (matching_output.fusion_weights.size(-1) == 4):
                    fusion_weights_4m = matching_output.fusion_weights
        scores = torch.concat(scores, dim=-1)
        text_scores = torch.concat(text_scores, dim=-1)
        image_scores = torch.concat(image_scores, dim=-1)
        cross_scores = torch.concat(cross_scores, dim=-1)
        triple_scores_tensor = torch.concat(triple_scores, dim=-1) if triple_scores else None
        struct_scores_tensor = torch.concat(struct_scores, dim=-1) if struct_scores else None
        if struct_scores_tensor is None and triple_scores_tensor is not None:
            struct_scores_tensor = triple_scores_tensor
        if triple_scores_tensor is None and struct_scores_tensor is not None:
            triple_scores_tensor = struct_scores_tensor
        if struct_scores_tensor is not None and hasattr(self.matching_module, 'arbitrate_full_scores'):
            full_matching_output = self.matching_module.arbitrate_full_scores(mention_features=mention_features, text_score=text_scores, image_score=image_scores, cross_score=cross_scores, struct_score=struct_scores_tensor)
            scores = full_matching_output.overall_score
            fusion_weights_4m = full_matching_output.fusion_weights_4m
            if fusion_weights_4m is None:
                fusion_weights_4m = full_matching_output.fusion_weights
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        target_ranks = None
        if answer is not None:
            answer = answer.to(device)
            target_ranks = rank[torch.arange(len(answer), device=rank.device), answer]
        return RetrievalOutput(scores=scores, text_scores=text_scores, image_scores=image_scores, cross_scores=cross_scores, triple_scores=triple_scores_tensor, struct_scores=struct_scores_tensor, rank=rank, target_ranks=target_ranks, fusion_weights_4m=fusion_weights_4m)
