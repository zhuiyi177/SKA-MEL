import math

import torch

import torch.nn as nn

import torch.nn.functional as F



from codes.backbone.moe import SwitchMoE

from codes.backbone.outputs import MatchingOutput, PerceptionOutput

from codes.backbone.triple_module import EntityTripleEncoder, MentionTripleEncoder, TripleIntraModalMatch



def _cfg_value(args, name: str, default):
    return getattr(getattr(args, 'model', object()), name, default)



def _extract_query_fusion_features(mention_text_cls: torch.Tensor, mention_image_cls: torch.Tensor, mention_image_tokens: torch.Tensor, text_proj: nn.Linear, image_proj: nn.Linear):
    mention_has_image = (mention_image_tokens.abs().sum(dim=(1, 2)) > 0).float().unsqueeze(-1)
    text_hidden = text_proj(mention_text_cls)
    image_hidden = image_proj(mention_image_cls)
    text_image_alignment = F.cosine_similarity(text_hidden, image_hidden, dim=-1).unsqueeze(-1)
    text_norm = mention_text_cls.norm(dim=-1, keepdim=True).log1p()
    image_norm = mention_image_cls.norm(dim=-1, keepdim=True).log1p()
    features = torch.cat([text_hidden, image_hidden, mention_has_image, text_norm, image_norm, text_image_alignment], dim=-1)
    return (features, mention_has_image)



class TextUnit(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.TGLU_hidden_dim)
        self.moe_layer = SwitchMoE(dim=self.args.model.input_hidden_dim, output_dim=self.args.model.TGLU_hidden_dim, num_experts=self.args.model.num_experts, top_k=self.args.model.top_experts)

    def forward(self, entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens):
        entity_text_cls_tokens_features = torch.cat([entity_text_cls.unsqueeze(dim=1), entity_text_tokens], dim=1)
        entity_text_moe = self.moe_layer(entity_text_cls_tokens_features, entity_text_cls_tokens_features)
        entity_text_cls_moe = entity_text_moe[:, 0, :]
        entity_cls_fc = entity_text_cls_moe.unsqueeze(dim=1)
        entity_text_tokens = entity_text_moe[:, 1:, :]
        mention_text_cls_tokens_features = torch.cat([mention_text_cls.unsqueeze(dim=1), mention_text_tokens], dim=1)
        mention_text_moe = self.moe_layer(mention_text_cls_tokens_features, mention_text_cls_tokens_features)
        mention_text_tokens = mention_text_moe[:, 1:, :]
        query = self.fc_query(entity_text_tokens)
        key = self.fc_key(mention_text_tokens)
        value = self.fc_value(mention_text_tokens)
        query = query.unsqueeze(dim=1)
        key = key.unsqueeze(dim=0)
        value = value.unsqueeze(dim=0)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.args.model.TGLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_probs, value)
        context = torch.mean(context, dim=-2)
        context = self.layer_norm(context)
        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)
        g2l_matching_score = g2l_matching_score.transpose(0, 1)
        g2g_matching_score = torch.matmul(mention_text_cls, entity_text_cls.transpose(-1, -2))
        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score



class ChannelEnhanceBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.fc_avg = nn.Linear(dim, dim)
        self.fc_max = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        gate = torch.sigmoid(self.fc_avg(avg_pool) + self.fc_max(max_pool))
        out = x * gate.unsqueeze(1)
        return self.layer_norm(out + x)



class VisionUnit(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.IDLU_hidden_dim)
        self.ceb = ChannelEnhanceBlock(self.args.model.IDLU_hidden_dim)
        self.moe_layer = SwitchMoE(dim=self.args.model.dv, output_dim=self.args.model.IDLU_hidden_dim, num_experts=self.args.model.num_experts, top_k=self.args.model.top_experts)

    def forward(self, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens):
        entity_image_cls_tokens_features = torch.cat([entity_image_cls.unsqueeze(dim=1), entity_image_tokens], dim=1)
        entity_image_moe = self.moe_layer(entity_image_cls_tokens_features, entity_image_cls_tokens_features)
        entity_image_cls_moe = entity_image_moe[:, 0, :]
        entity_cls_fc = entity_image_cls_moe.unsqueeze(dim=1)
        entity_image_tokens = entity_image_moe[:, 1:, :]
        entity_image_tokens = self.ceb(entity_image_tokens)
        mention_image_cls_tokens_features = torch.cat([mention_image_cls.unsqueeze(dim=1), mention_image_tokens], dim=1)
        mention_image_moe = self.moe_layer(mention_image_cls_tokens_features, mention_image_cls_tokens_features)
        mention_image_tokens = mention_image_moe[:, 1:, :]
        mention_image_tokens = self.ceb(mention_image_tokens)
        query = self.fc_query(entity_image_tokens)
        key = self.fc_key(mention_image_tokens)
        value = self.fc_value(mention_image_tokens)
        query = query.unsqueeze(dim=1)
        key = key.unsqueeze(dim=0)
        value = value.unsqueeze(dim=0)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.args.model.IDLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_probs, value)
        context = torch.mean(context, dim=-2)
        context = self.layer_norm(context)
        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)
        g2l_matching_score = g2l_matching_score.transpose(0, 1)
        g2g_matching_score = torch.matmul(mention_image_cls, entity_image_cls.transpose(-1, -2))
        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score



class CrossUnit(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.CMFU_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.CMFU_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)
        self.context_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)
        self.moe_layer = SwitchMoE(dim=self.args.model.CMFU_hidden_dim, output_dim=self.args.model.CMFU_hidden_dim, num_experts=self.args.model.num_experts, top_k=self.args.model.top_experts)

    def forward(self, entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens):
        entity_text_cls = self.text_fc(entity_text_cls)
        entity_text_cls_ori = entity_text_cls
        entity_text_tokens = self.text_fc(entity_text_tokens)
        mention_text_cls = self.text_fc(mention_text_cls)
        mention_text_cls_ori = mention_text_cls
        mention_text_tokens = self.text_fc(mention_text_tokens)
        entity_image_cls = self.image_fc(entity_image_cls)
        entity_image_cls_ori = entity_image_cls
        entity_image_tokens = self.image_fc(entity_image_tokens)
        mention_image_cls = self.image_fc(mention_image_cls)
        mention_image_cls_ori = mention_image_cls
        mention_image_tokens = self.image_fc(mention_image_tokens)
        entity_text_cls_image_tokens_features = torch.cat([entity_text_cls.unsqueeze(dim=1), entity_image_tokens], dim=1)
        entity_image_cls_text_tokens_features = torch.cat([entity_image_cls.unsqueeze(dim=1), entity_text_tokens], dim=1)
        mention_text_cls_image_tokens_features = torch.cat([mention_text_cls.unsqueeze(dim=1), mention_image_tokens], dim=1)
        mention_image_cls_text_tokens_features = torch.cat([mention_image_cls.unsqueeze(dim=1), mention_text_tokens], dim=1)
        entity_text_cls_image_tokens_moe = self.moe_layer(entity_text_cls_image_tokens_features, entity_image_cls_text_tokens_features)
        entity_text_cls = entity_text_cls_image_tokens_moe[:, 0, :]
        entity_image_tokens = entity_text_cls_image_tokens_moe[:, 1:, :]
        mention_text_cls_image_tokens_moe = self.moe_layer(mention_text_cls_image_tokens_features, mention_image_cls_text_tokens_features)
        mention_text_cls = mention_text_cls_image_tokens_moe[:, 0, :]
        mention_image_tokens = mention_text_cls_image_tokens_moe[:, 1:, :]
        entity_image_cls_text_tokens_moe = self.moe_layer(entity_image_cls_text_tokens_features, entity_text_cls_image_tokens_features)
        entity_image_cls = entity_image_cls_text_tokens_moe[:, 0, :]
        entity_text_tokens = entity_image_cls_text_tokens_moe[:, 1:, :]
        mention_image_cls_text_tokens_moe = self.moe_layer(mention_image_cls_text_tokens_features, mention_text_cls_image_tokens_features)
        mention_image_cls = mention_image_cls_text_tokens_moe[:, 0, :]
        mention_text_tokens = mention_image_cls_text_tokens_moe[:, 1:, :]
        entity_text_cls = entity_text_cls.unsqueeze(dim=1)
        entity_text_image_cross_modal_score = torch.matmul(entity_text_cls, entity_image_tokens.transpose(-1, -2))
        entity_text_image_cross_modal_probs = nn.Softmax(dim=-1)(entity_text_image_cross_modal_score)
        entity_text_image_context = torch.matmul(entity_text_image_cross_modal_probs, entity_image_tokens).squeeze()
        entity_text_image_gate = torch.sigmoid(self.gate_fc(entity_text_cls_ori + entity_text_image_context))
        entity_text_image_context = self.gate_layer_norm(entity_text_cls_ori * entity_text_image_gate + entity_text_image_context * (1 - entity_text_image_gate))
        mention_text_cls = mention_text_cls.unsqueeze(dim=1)
        mention_text_image_cross_modal_score = torch.matmul(mention_text_cls, mention_image_tokens.transpose(-1, -2))
        mention_text_image_cross_modal_probs = nn.Softmax(dim=-1)(mention_text_image_cross_modal_score)
        mention_text_image_context = torch.matmul(mention_text_image_cross_modal_probs, mention_image_tokens).squeeze()
        mention_text_image_gate = torch.sigmoid(self.gate_fc(mention_text_cls_ori + mention_text_image_context))
        mention_text_image_context = self.gate_layer_norm(mention_text_cls_ori * mention_text_image_gate + mention_text_image_context * (1 - mention_text_image_gate))
        score_text_image = torch.matmul(mention_text_image_context, entity_text_image_context.transpose(-1, -2))
        entity_image_cls = entity_image_cls.unsqueeze(dim=1)
        entity_image_text_cross_modal_score = torch.matmul(entity_image_cls, entity_text_tokens.transpose(-1, -2))
        entity_image_text_cross_modal_probs = nn.Softmax(dim=-1)(entity_image_text_cross_modal_score)
        entity_image_text_context = torch.matmul(entity_image_text_cross_modal_probs, entity_text_tokens).squeeze()
        entity_image_text_gate = torch.sigmoid(self.gate_fc(entity_image_cls_ori + entity_image_text_context))
        entity_image_text_context = self.gate_layer_norm(entity_image_cls_ori * entity_image_text_gate + entity_image_text_context * (1 - entity_image_text_gate))
        mention_image_cls = mention_image_cls.unsqueeze(dim=1)
        mention_image_text_cross_modal_score = torch.matmul(mention_image_cls, mention_text_tokens.transpose(-1, -2))
        mention_image_text_cross_modal_probs = nn.Softmax(dim=-1)(mention_image_text_cross_modal_score)
        mention_image_text_context = torch.matmul(mention_image_text_cross_modal_probs, mention_text_tokens).squeeze()
        mention_image_text_gate = torch.sigmoid(self.gate_fc(mention_image_cls_ori + mention_image_text_context))
        mention_image_text_context = self.gate_layer_norm(mention_image_cls_ori * mention_image_text_gate + mention_image_text_context * (1 - mention_image_text_gate))
        score_image_text = torch.matmul(mention_image_text_context, entity_image_text_context.transpose(-1, -2))
        score = (score_text_image + score_image_text) / 2
        return score



class MatchingModule(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_module = TextUnit(self.args)
        self.vision_module = VisionUnit(self.args)
        self.cross_module = CrossUnit(self.args)
        self.text_cls_layernorm = nn.LayerNorm(self.args.model.dt)
        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.dt)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.dv)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.dv)

    def _coerce_inputs(self, *args):
        if len(args) == 2 and all((isinstance(item, PerceptionOutput) for item in args)):
            entity_features, mention_features = args
            return (entity_features.text_embeds, entity_features.text_seq_tokens, mention_features.text_embeds, mention_features.text_seq_tokens, entity_features.image_embeds, entity_features.image_patch_tokens, mention_features.image_embeds, mention_features.image_patch_tokens)
        if len(args) == 8:
            return args
        raise TypeError('MatchingModule.forward expects two PerceptionOutput objects or eight explicit tensors.')

    def _normalize_modal_inputs(self, entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens):
        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_text_cls = self.text_cls_layernorm(mention_text_cls)
        entity_text_tokens = self.text_tokens_layernorm(entity_text_tokens)
        mention_text_tokens = self.text_tokens_layernorm(mention_text_tokens)
        entity_image_cls = self.image_cls_layernorm(entity_image_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)
        entity_image_tokens = self.image_tokens_layernorm(entity_image_tokens)
        mention_image_tokens = self.image_tokens_layernorm(mention_image_tokens)
        return (entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens)

    def _compute_three_branch_scores(self, *args):
        entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens = self._normalize_modal_inputs(*self._coerce_inputs(*args))
        text_matching_score = self.text_module(entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens)
        image_matching_score = self.vision_module(entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens)
        image_text_matching_score = self.cross_module(entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens)
        return (text_matching_score, image_matching_score, image_text_matching_score, mention_text_cls, mention_image_cls, mention_image_tokens)

    def forward(self, *args) -> MatchingOutput:
        text_matching_score, image_matching_score, image_text_matching_score, _, _, _ = self._compute_three_branch_scores(*args)
        score = (text_matching_score + image_matching_score + image_text_matching_score) / 3
        return MatchingOutput(overall_score=score, text_score=text_matching_score, image_score=image_matching_score, cross_score=image_text_matching_score)



class BranchScoreCalibration4M(nn.Module):
    def __init__(self, args):
        super().__init__()
        init_scale = torch.tensor(
            list(_cfg_value(args, "fusion_init_scale_4m", [1.0, 1.0, 1.0, 1.0])),
            dtype=torch.float32,
        )
        if init_scale.numel() != 4:
            init_scale = torch.ones(4, dtype=torch.float32)
        self.log_scale = nn.Parameter(torch.log(init_scale.clamp_min(1e-4)))

    def forward(self, text_score, image_score, cross_score, struct_score):
        scale = torch.exp(self.log_scale)
        return (
            scale[0] * text_score,
            scale[1] * image_score,
            scale[2] * cross_score,
            scale[3] * struct_score,
        )



class DatasetPriorCalibration4M(nn.Module):

    def __init__(self, args):
        super().__init__()
        fusion_prior = torch.tensor(list(_cfg_value(args, 'fusion_prior_4m', [0.45, 0.15, 0.3, 0.1])), dtype=torch.float32)
        if fusion_prior.numel() != 4:
            fusion_prior = torch.tensor([0.45, 0.15, 0.3, 0.1], dtype=torch.float32)
        fusion_prior = fusion_prior / fusion_prior.sum().clamp_min(1e-06)
        init_delta_scale = float(_cfg_value(args, 'dataset_prior_delta_scale_init_4m', _cfg_value(args, 'dataset_prior_delta_scale_init', 1.0)))
        max_delta_scale = _cfg_value(args, 'dataset_prior_delta_scale_max_4m', None)
        self.prior_logits = nn.Parameter(torch.log(fusion_prior.clamp_min(0.0001)), requires_grad=False)
        self.delta_scale = nn.Parameter(torch.tensor(init_delta_scale, dtype=torch.float32))
        self.delta_scale_max = None if max_delta_scale is None else float(max_delta_scale)

    def forward(self, delta_logits: torch.Tensor) -> torch.Tensor:
        delta_scale = self.delta_scale.clamp_min(0.0)
        if self.delta_scale_max is not None:
            delta_scale = delta_scale.clamp(max=self.delta_scale_max)
        return self.prior_logits.unsqueeze(0) + delta_scale * delta_logits



class PriorAnchoredQueryAwareFusionHead4M(nn.Module):
    def __init__(self, args):
        super().__init__()
        fusion_hidden_dim = max(8, int(_cfg_value(args, "fusion_hidden_dim_4m", 64)))
        self.max_logit_delta = float(_cfg_value(args, "fusion_max_logit_delta_4m", 0.10))
        self.dataset_prior = DatasetPriorCalibration4M(args)
        self.text_proj = nn.Linear(int(_cfg_value(args, "dt", 512)), fusion_hidden_dim)
        self.image_proj = nn.Linear(int(_cfg_value(args, "dv", 128)), fusion_hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(fusion_hidden_dim * 2 + 4),
            nn.Linear(fusion_hidden_dim * 2 + 4, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, 4),
        )

    def forward(self, mention_text_cls, mention_image_cls, mention_image_tokens):
        features, _ = _extract_query_fusion_features(
            mention_text_cls=mention_text_cls,
            mention_image_cls=mention_image_cls,
            mention_image_tokens=mention_image_tokens,
            text_proj=self.text_proj,
            image_proj=self.image_proj,
        )
        delta_logits = torch.tanh(self.network(features)) * self.max_logit_delta
        logits = self.dataset_prior(delta_logits)
        return torch.softmax(logits, dim=-1)



class StructuralModalityMixin:
    def _setup_structural_modality(self, args) -> None:
        self.use_triple = bool(_cfg_value(args, "use_triple", True))
        if self.use_triple:
            self.entity_triple_encoder = EntityTripleEncoder(args)
            self.mention_triple_encoder = MentionTripleEncoder(args)
            self.triple_match = TripleIntraModalMatch(args)

    def encode_entity_triples(self, clip_text_model, triple_input_ids, triple_attention_mask, triple_valid_mask):
        if not self.use_triple:
            return None, None, None
        if triple_input_ids is None or triple_attention_mask is None or triple_valid_mask is None:
            return None, None, None
        entity_triple_cls, entity_triple_tokens = self.entity_triple_encoder(
            clip_text_model, triple_input_ids, triple_attention_mask, triple_valid_mask
        )
        return entity_triple_cls, entity_triple_tokens, triple_valid_mask

    def build_mention_triples(self, mention_features: PerceptionOutput) -> PerceptionOutput:
        if not self.use_triple:
            return mention_features
        if mention_features.triple_cls is not None and mention_features.triple_tokens is not None:
            return mention_features
        mention_triple_cls, mention_triple_tokens = self.mention_triple_encoder(
            mention_features.text_seq_tokens, mention_features.image_patch_tokens
        )
        return mention_features.with_triple(
            triple_cls=mention_triple_cls,
            triple_tokens=mention_triple_tokens,
            triple_valid_mask=None,
        )



class SKAMELMatchingModule(StructuralModalityMixin, MatchingModule):

    def __init__(self, args):
        MatchingModule.__init__(self, args)
        self._setup_structural_modality(args)
        self.branch_calibration_4m = BranchScoreCalibration4M(args)
        self.fusion_head_4m = PriorAnchoredQueryAwareFusionHead4M(args)

    @staticmethod
    def _build_three_modal_output(text_score, image_score, cross_score):
        overall_score = (text_score + image_score + cross_score) / 3
        return MatchingOutput(overall_score=overall_score, text_score=text_score, image_score=image_score, cross_score=cross_score)

    def forward(self, *args) -> MatchingOutput:
        text_score, image_score, cross_score, mention_text_cls, mention_image_cls, mention_image_tokens = self._compute_three_branch_scores(*args)
        if not self.use_triple:
            return self._build_three_modal_output(text_score, image_score, cross_score)
        if not (len(args) == 2 and all((isinstance(item, PerceptionOutput) for item in args))):
            return self._build_three_modal_output(text_score, image_score, cross_score)
        entity_features, mention_features = args
        mention_features = self.build_mention_triples(mention_features)
        entity_triple_cls = entity_features.triple_cls
        entity_triple_tokens = entity_features.triple_tokens
        entity_triple_valid_mask = entity_features.triple_valid_mask
        mention_triple_cls = mention_features.triple_cls
        mention_triple_tokens = mention_features.triple_tokens
        if entity_triple_cls is None or entity_triple_tokens is None or mention_triple_cls is None or (mention_triple_tokens is None):
            return self._build_three_modal_output(text_score, image_score, cross_score)
        struct_score = self.triple_match(entity_triple_cls, entity_triple_tokens, mention_triple_cls, mention_triple_tokens, entity_triple_valid_mask=entity_triple_valid_mask)
        calibrated_text, calibrated_image, calibrated_cross, calibrated_struct = self.branch_calibration_4m(text_score, image_score, cross_score, struct_score)
        fusion_weights_4m = self.fusion_head_4m(mention_text_cls=mention_text_cls, mention_image_cls=mention_image_cls, mention_image_tokens=mention_image_tokens)
        overall_score = fusion_weights_4m[:, 0:1] * calibrated_text + fusion_weights_4m[:, 1:2] * calibrated_image + fusion_weights_4m[:, 2:3] * calibrated_cross + fusion_weights_4m[:, 3:4] * calibrated_struct
        return MatchingOutput(overall_score=overall_score, text_score=text_score, image_score=image_score, cross_score=cross_score, triple_score=struct_score, struct_score=struct_score, fusion_weights=fusion_weights_4m, fusion_weights_4m=fusion_weights_4m)
