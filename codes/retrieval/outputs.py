from dataclasses import dataclass
from typing import Optional
import torch
from codes.backbone.outputs import PerceptionOutput

@dataclass
class EntityCache:
    entity_text_embeds: torch.Tensor
    entity_image_embeds: torch.Tensor
    entity_text_seq_tokens: torch.Tensor
    entity_image_patch_tokens: torch.Tensor
    entity_triple_cls: Optional[torch.Tensor] = None
    entity_triple_tokens: Optional[torch.Tensor] = None
    entity_triple_valid_mask: Optional[torch.Tensor] = None

    def slice(self, start_pos: int, end_pos: int, device=None) -> PerceptionOutput:
        return PerceptionOutput(text_embeds=self.entity_text_embeds[start_pos:end_pos].to(device), image_embeds=self.entity_image_embeds[start_pos:end_pos].to(device), text_seq_tokens=self.entity_text_seq_tokens[start_pos:end_pos].to(device), image_patch_tokens=self.entity_image_patch_tokens[start_pos:end_pos].to(device), triple_cls=self.entity_triple_cls[start_pos:end_pos].to(device) if self.entity_triple_cls is not None else None, triple_tokens=self.entity_triple_tokens[start_pos:end_pos].to(device) if self.entity_triple_tokens is not None else None, triple_valid_mask=self.entity_triple_valid_mask[start_pos:end_pos].to(device) if self.entity_triple_valid_mask is not None else None)

@dataclass
class RetrievalOutput:
    scores: torch.Tensor
    text_scores: torch.Tensor
    image_scores: torch.Tensor
    cross_scores: torch.Tensor
    rank: torch.Tensor
    triple_scores: Optional[torch.Tensor] = None
    struct_scores: Optional[torch.Tensor] = None
    target_ranks: Optional[torch.Tensor] = None
    fusion_weights_4m: Optional[torch.Tensor] = None
