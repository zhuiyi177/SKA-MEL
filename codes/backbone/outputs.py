from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class PerceptionOutput:
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    text_seq_tokens: torch.Tensor
    image_patch_tokens: torch.Tensor
    triple_cls: Optional[torch.Tensor] = None
    triple_tokens: Optional[torch.Tensor] = None
    triple_valid_mask: Optional[torch.Tensor] = None

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.text_embeds, self.image_embeds, self.text_seq_tokens, self.image_patch_tokens)

    def with_triple(self, triple_cls: Optional[torch.Tensor], triple_tokens: Optional[torch.Tensor], triple_valid_mask: Optional[torch.Tensor]=None) -> 'PerceptionOutput':
        return PerceptionOutput(text_embeds=self.text_embeds, image_embeds=self.image_embeds, text_seq_tokens=self.text_seq_tokens, image_patch_tokens=self.image_patch_tokens, triple_cls=triple_cls, triple_tokens=triple_tokens, triple_valid_mask=triple_valid_mask)

@dataclass
class MatchingOutput:
    overall_score: torch.Tensor
    text_score: torch.Tensor
    image_score: torch.Tensor
    cross_score: torch.Tensor
    triple_score: Optional[torch.Tensor] = None
    struct_score: Optional[torch.Tensor] = None
    fusion_weights: Optional[torch.Tensor] = None
    fusion_weights_4m: Optional[torch.Tensor] = None

    def as_tuple(self):
        return (self.overall_score, (self.text_score, self.image_score, self.cross_score))
