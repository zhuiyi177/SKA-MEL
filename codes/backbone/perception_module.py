import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from codes.backbone.outputs import PerceptionOutput
from codes.utils.functions import get_project_root, resolve_project_path

class FrequencyEnhanceBlock(nn.Module):
    MAX_PHASE_SHIFT = math.pi / 4.0

    def __init__(self, dim: int, max_gate: float=0.3):
        super().__init__()
        self.amp_fc = nn.Linear(dim, dim)
        self.phase_fc = nn.Linear(dim, dim)
        self.max_gate = max_gate
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            amp_w = self.amp_fc.weight.float()
            amp_b = self.amp_fc.bias.float()
            pha_w = self.phase_fc.weight.float()
            pha_b = self.phase_fc.bias.float()
            x_fft = torch.fft.rfft(xf, dim=1)
            amp = x_fft.abs()
            phase = x_fft.angle()
            amp_scale = 1.0 + 0.5 * torch.tanh(F.linear(amp, amp_w, amp_b))
            amp_enh = amp_scale * amp
            phase_shift = torch.tanh(F.linear(phase, pha_w, pha_b)) * self.MAX_PHASE_SHIFT
            phase_enh = phase + phase_shift
            x_fft_enh = torch.polar(amp_enh, phase_enh)
            x_out = torch.fft.irfft(x_fft_enh, n=xf.shape[1], dim=1)
            xf_norm = xf.norm(dim=-1, keepdim=True).clamp(min=1e-06)
            x_out_norm = x_out.norm(dim=-1, keepdim=True).clamp(min=1e-06)
            x_out = x_out * (xf_norm / x_out_norm)
            gate = self.max_gate * torch.clamp(torch.tanh(self.gate), min=0.0)
            result = xf + gate * (x_out - xf)
        return result.to(orig_dtype)

class PerceptionModule(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.project_root = getattr(self.args, 'project_root', get_project_root())
        self.pretrained_model_path = resolve_project_path(self.project_root, self.args.pretrained_model)
        self.clip = CLIPModel.from_pretrained(self.pretrained_model_path)
        self.image_cls_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.dv)
        self.image_tokens_fc = nn.Linear(self.args.model.input_image_hidden_dim, self.args.model.dv)
        self._enable_feb = getattr(self.args.model, 'enable_feb', True)
        if self._enable_feb:
            feb_max_gate = getattr(self.args.model, 'feb_max_gate', 0.3)
            self.feb = FrequencyEnhanceBlock(self.args.model.dv, max_gate=feb_max_gate)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None) -> PerceptionOutput:
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_embeds = clip_output.text_embeds
        image_embeds = clip_output.image_embeds
        text_seq_tokens = clip_output.text_model_output[0]
        image_patch_tokens = clip_output.vision_model_output[0]
        image_embeds = self.image_cls_fc(image_embeds)
        image_patch_tokens = self.image_tokens_fc(image_patch_tokens)
        if self._enable_feb:
            cls_tok = image_patch_tokens[:, :1, :]
            patch_tok = image_patch_tokens[:, 1:, :]
            patch_tok = self.feb(patch_tok)
            image_patch_tokens = torch.cat([cls_tok, patch_tok], dim=1)
        return PerceptionOutput(text_embeds=text_embeds, image_embeds=image_embeds, text_seq_tokens=text_seq_tokens, image_patch_tokens=image_patch_tokens)
