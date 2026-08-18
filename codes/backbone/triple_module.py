import math
import torch
import torch.nn as nn

def _cfg_value(args, name: str, default):
    return getattr(getattr(args, 'model', object()), name, default)

class EntityTripleEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.triple_hidden_dim = int(_cfg_value(args, 'triple_hidden_dim', 96))
        self.input_hidden_dim = int(_cfg_value(args, 'input_hidden_dim', _cfg_value(args, 'dt', 512)))
        self.triple_proj = nn.Linear(self.input_hidden_dim, self.triple_hidden_dim)
        self.triple_cls_layernorm = nn.LayerNorm(self.triple_hidden_dim)
        self.triple_tokens_layernorm = nn.LayerNorm(self.triple_hidden_dim)

    def forward(self, clip_text_model, triple_input_ids, triple_attention_mask, triple_valid_mask):
        batch_size, triple_num, seq_len = triple_input_ids.size()
        flat_input_ids = triple_input_ids.view(batch_size * triple_num, seq_len)
        flat_attention_mask = triple_attention_mask.view(batch_size * triple_num, seq_len)
        text_outputs = clip_text_model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        flat_text_tokens = text_outputs[0]
        token_mask = flat_attention_mask.unsqueeze(-1).float()
        token_denom = token_mask.sum(dim=1).clamp(min=1.0)
        flat_text_pooled = (flat_text_tokens * token_mask).sum(dim=1) / token_denom
        flat_triple_tokens = self.triple_proj(flat_text_pooled)
        entity_triple_tokens = flat_triple_tokens.view(batch_size, triple_num, self.triple_hidden_dim)
        valid_mask = triple_valid_mask.float().unsqueeze(-1)
        entity_triple_tokens = entity_triple_tokens * valid_mask
        entity_triple_tokens = self.triple_tokens_layernorm(entity_triple_tokens)
        entity_triple_tokens = entity_triple_tokens * valid_mask
        valid_mask_2d = triple_valid_mask.float()
        valid_denom = valid_mask_2d.sum(dim=1, keepdim=True).clamp(min=1.0)
        entity_triple_cls = (entity_triple_tokens * valid_mask_2d.unsqueeze(-1)).sum(dim=1) / valid_denom
        entity_triple_cls = self.triple_cls_layernorm(entity_triple_cls)
        return (entity_triple_cls, entity_triple_tokens)

class MentionTripleEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.triple_hidden_dim = int(_cfg_value(args, 'triple_hidden_dim', 96))
        self.triple_slot_num = int(_cfg_value(args, 'triple_slot_num', 4))
        text_dim = int(_cfg_value(args, 'dt', 512))
        image_dim = int(_cfg_value(args, 'dv', 128))
        self.text_proj = nn.Linear(text_dim, self.triple_hidden_dim)
        self.image_proj = nn.Linear(image_dim, self.triple_hidden_dim)
        joint_dim = self.triple_hidden_dim * 2
        self.cls_mlp = nn.Sequential(nn.Linear(joint_dim, joint_dim), nn.GELU(), nn.Linear(joint_dim, self.triple_hidden_dim))
        self.slot_mlp = nn.Sequential(nn.Linear(joint_dim, joint_dim), nn.GELU(), nn.Linear(joint_dim, self.triple_slot_num * self.triple_hidden_dim))
        self.triple_cls_layernorm = nn.LayerNorm(self.triple_hidden_dim)
        self.triple_tokens_layernorm = nn.LayerNorm(self.triple_hidden_dim)

    def forward(self, mention_text_tokens, mention_image_tokens):
        text_proj = self.text_proj(mention_text_tokens)
        image_proj = self.image_proj(mention_image_tokens)
        mention_text_pool = torch.mean(text_proj, dim=1)
        mention_image_pool = torch.mean(image_proj, dim=1)
        joint = torch.cat([mention_text_pool, mention_image_pool], dim=-1)
        mention_triple_cls = self.cls_mlp(joint)
        mention_triple_cls = self.triple_cls_layernorm(mention_triple_cls)
        mention_triple_tokens = self.slot_mlp(joint).view(-1, self.triple_slot_num, self.triple_hidden_dim)
        mention_triple_tokens = self.triple_tokens_layernorm(mention_triple_tokens)
        return (mention_triple_cls, mention_triple_tokens)

class TripleIntraModalMatch(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.triple_hidden_dim = int(_cfg_value(args, 'triple_hidden_dim', 96))
        self.triple_matching_mode = str(_cfg_value(args, 'triple_matching_mode', 'dual')).strip().lower()
        valid_matching_modes = {'dual', 'global_only', 'local_only'}
        if self.triple_matching_mode not in valid_matching_modes:
            raise ValueError(f'Unsupported triple_matching_mode={self.triple_matching_mode!r}; expected one of {sorted(valid_matching_modes)}.')
        self.fc_query = nn.Linear(self.triple_hidden_dim, self.triple_hidden_dim)
        self.fc_key = nn.Linear(self.triple_hidden_dim, self.triple_hidden_dim)
        self.fc_value = nn.Linear(self.triple_hidden_dim, self.triple_hidden_dim)
        self.fc_cls = nn.Linear(self.triple_hidden_dim, self.triple_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.triple_hidden_dim)

    def forward(self, entity_triple_cls, entity_triple_tokens, mention_triple_cls, mention_triple_tokens, entity_triple_valid_mask=None):
        if entity_triple_valid_mask is None:
            entity_triple_valid_mask = entity_triple_tokens.new_ones(entity_triple_tokens.shape[:2])
        triple_token_mask = entity_triple_valid_mask.float().unsqueeze(dim=1).unsqueeze(-1)
        entity_has_valid = (entity_triple_valid_mask.sum(dim=-1, keepdim=True) > 0).float()
        entity_cls_fc = (self.fc_cls(entity_triple_cls) * entity_has_valid).unsqueeze(dim=1)
        query = self.fc_query(entity_triple_tokens).unsqueeze(dim=1) * triple_token_mask
        key = self.fc_key(mention_triple_tokens).unsqueeze(dim=0)
        value = self.fc_value(mention_triple_tokens).unsqueeze(dim=0)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.triple_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_probs, value)
        valid_denom = triple_token_mask.sum(dim=-2).clamp(min=1.0)
        context = (context * triple_token_mask).sum(dim=-2) / valid_denom
        context = self.layer_norm(context)
        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)
        g2l_matching_score = g2l_matching_score.transpose(0, 1)
        g2g_matching_score = torch.matmul(mention_triple_cls, entity_triple_cls.transpose(-1, -2))
        if self.triple_matching_mode == 'global_only':
            return g2g_matching_score
        if self.triple_matching_mode == 'local_only':
            return g2l_matching_score
        return (g2l_matching_score + g2g_matching_score) / 2
