import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from codes.backbone.matching_module import SKAMELMatchingModule
from codes.backbone.perception_module import PerceptionModule
from codes.retrieval.retrieval_agent import RetrievalAgent

def build_matching_module(args):
    return SKAMELMatchingModule(args)

class SKAMELModel(pl.LightningModule):
    _IGNORED_CHECKPOINT_KEYS = {
        'matcher.triple_fusion_beta',
        'matcher.branch_calibration_4m.bias',
    }

    def __init__(self, args):
        super(SKAMELModel, self).__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.encoder = PerceptionModule(args)
        self.matcher = build_matching_module(args)
        self._print_fusion_prior_4m_if_configured()
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.lambda_tri = float(getattr(getattr(self.args, 'model', object()), 'lambda_tri', 1.0))
        self.retrieval_agent = RetrievalAgent(perception_module=self.encoder, matching_module=self.matcher, num_entity=int(self.args.data.num_entity))
        self.entity_cache = None
        self.entity_text_embeds = None
        self.entity_image_embeds = None
        self.entity_text_seq_tokens = None
        self.entity_image_patch_tokens = None
        self.entity_triple_cls = None
        self.entity_triple_tokens = None
        self.entity_triple_valid_mask = None
        distill_cfg = getattr(self.args, 'distillation', None)
        self.distillation_enabled = bool(getattr(distill_cfg, 'enable', False))
        self.distillation_temperature = float(getattr(distill_cfg, 'temperature', 4.0))
        self.distillation_alpha = float(getattr(distill_cfg, 'alpha', 2.25))
        self.teacher_start_epoch = int(getattr(distill_cfg, 'teacher_start_epoch', 1))
        self.refresh_teacher_every_epoch = bool(getattr(distill_cfg, 'refresh_teacher_every_epoch', True))
        self.teacher_ckpt_path = ''
        self.teacher_encoder = None
        self.teacher_matcher = None

    @property
    def perception_module(self):
        return self.encoder

    @property
    def matching_module(self):
        return self.matcher

    def set_entity_cache(self, entity_cache) -> None:
        self.entity_cache = entity_cache
        self.retrieval_agent.set_entity_cache(entity_cache)
        if entity_cache is None:
            self.entity_text_embeds = None
            self.entity_image_embeds = None
            self.entity_text_seq_tokens = None
            self.entity_image_patch_tokens = None
            self.entity_triple_cls = None
            self.entity_triple_tokens = None
            self.entity_triple_valid_mask = None
            return
        self.entity_text_embeds = entity_cache.entity_text_embeds
        self.entity_image_embeds = entity_cache.entity_image_embeds
        self.entity_text_seq_tokens = entity_cache.entity_text_seq_tokens
        self.entity_image_patch_tokens = entity_cache.entity_image_patch_tokens
        self.entity_triple_cls = entity_cache.entity_triple_cls
        self.entity_triple_tokens = entity_cache.entity_triple_tokens
        self.entity_triple_valid_mask = entity_cache.entity_triple_valid_mask

    def clear_entity_cache(self) -> None:
        self.set_entity_cache(None)

    @staticmethod
    def _drop_teacher_state_keys(state_dict) -> int:
        if state_dict is None:
            return 0
        removed = 0
        for key in list(state_dict.keys()):
            if key.startswith('teacher_encoder.') or key.startswith('teacher_matcher.'):
                state_dict.pop(key)
                removed += 1
        return removed

    def on_save_checkpoint(self, checkpoint) -> None:
        self._drop_teacher_state_keys(checkpoint.get('state_dict'))

    def load_state_dict(self, state_dict, strict: bool=True):
        state_dict = state_dict.copy()
        self._drop_teacher_state_keys(state_dict)
        for key in self._IGNORED_CHECKPOINT_KEYS:
            state_dict.pop(key, None)
        return super().load_state_dict(state_dict, strict=strict)

    def on_train_epoch_start(self) -> None:
        self._refresh_teacher_if_needed()

    def training_step(self, batch):
        ent_batch = {}
        mention_batch = {}
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v
        ent_batch.pop('empty_img_flag')
        ent_triple_input_ids = ent_batch.pop('triple_input_ids', None)
        ent_triple_attention_mask = ent_batch.pop('triple_attention_mask', None)
        ent_triple_valid_mask = ent_batch.pop('triple_valid_mask', None)
        mention_features = self.encoder(**mention_batch)
        entity_features = self.encoder(**ent_batch)
        if hasattr(self.matcher, 'encode_entity_triples'):
            entity_triple_cls, entity_triple_tokens, entity_triple_valid_mask = self.matcher.encode_entity_triples(clip_text_model=self.encoder.clip.text_model, triple_input_ids=ent_triple_input_ids, triple_attention_mask=ent_triple_attention_mask, triple_valid_mask=ent_triple_valid_mask)
            if entity_triple_cls is not None and entity_triple_tokens is not None:
                entity_features = entity_features.with_triple(triple_cls=entity_triple_cls, triple_tokens=entity_triple_tokens, triple_valid_mask=entity_triple_valid_mask)
        if hasattr(self.matcher, 'build_mention_triples'):
            mention_features = self.matcher.build_mention_triples(mention_features)
        matching_output = self.matcher(entity_features, mention_features)
        fusion_weights_4m = matching_output.fusion_weights_4m
        if fusion_weights_4m is None:
            fusion_weights = matching_output.fusion_weights
            if fusion_weights is not None and fusion_weights.dim() == 2 and (fusion_weights.size(-1) == 4):
                fusion_weights_4m = fusion_weights
        self._log_fusion_weights_4m_stats(fusion_weights_4m, prefix='Train')
        logits = matching_output.overall_score
        text_logits = matching_output.text_score
        image_logits = matching_output.image_score
        image_text_logits = matching_output.cross_score
        struct_logits = matching_output.struct_score if matching_output.struct_score is not None else matching_output.triple_score
        labels = torch.arange(len(mention_features.text_embeds)).long().to(mention_features.text_embeds.device)
        teacher_output = self._compute_teacher_matching_output(mention_batch=mention_batch, ent_batch=ent_batch, ent_triple_input_ids=ent_triple_input_ids, ent_triple_attention_mask=ent_triple_attention_mask, ent_triple_valid_mask=ent_triple_valid_mask)
        if teacher_output is not None:
            text_loss = self.compute_weighted_distillation_loss(student_logits=text_logits, teacher_logits=teacher_output.text_score, labels=labels)
            image_loss = self.compute_weighted_distillation_loss(student_logits=image_logits, teacher_logits=teacher_output.image_score, labels=labels)
            image_text_loss = self.compute_weighted_distillation_loss(student_logits=image_text_logits, teacher_logits=teacher_output.cross_score, labels=labels)
            overall_loss = self.compute_weighted_distillation_loss(student_logits=logits, teacher_logits=teacher_output.overall_score, labels=labels)
        else:
            text_loss = self.loss_fct(text_logits, labels)
            image_loss = self.loss_fct(image_logits, labels)
            image_text_loss = self.loss_fct(image_text_logits, labels)
            overall_loss = self.loss_fct(logits, labels)
        if struct_logits is not None:
            struct_loss = self.loss_fct(struct_logits, labels)
        else:
            struct_loss = logits.new_zeros(())
        loss = overall_loss + text_loss + image_loss + image_text_loss + self.lambda_tri * struct_loss
        self.log('Train/loss', loss.detach().cpu().item(), on_epoch=True, prog_bar=True)
        self.log('Train/loss_triple', struct_loss.detach().cpu().item(), on_epoch=True, prog_bar=False)
        self.log('Train/loss_struct', struct_loss.detach().cpu().item(), on_epoch=True, prog_bar=False)
        self.log('Train/distill_active', 1.0 if teacher_output is not None else 0.0, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        answer = batch.pop('answer')
        retrieval_output = self.retrieval_agent.retrieve(batch=batch, device=self.device, chunk_size=int(self.args.data.eval_chunk_size), answer=answer)
        self._log_fusion_weights_4m_stats(retrieval_output.fusion_weights_4m, prefix='Val')
        tgt_rank = retrieval_output.target_ranks.detach().cpu()
        return dict(rank=tgt_rank, all_rank=retrieval_output.rank.detach().cpu().numpy())

    def on_validation_start(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        entity_cache = self.retrieval_agent.build_entity_cache(entity_dataloader, self.device)
        self.set_entity_cache(entity_cache)

    def validation_epoch_end(self, outputs):
        self.clear_entity_cache()
        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()
        self.log('Val/hits20', hits20)
        self.log('Val/hits10', hits10)
        self.log('Val/hits5', hits5)
        self.log('Val/hits3', hits3)
        self.log('Val/hits1', hits1)
        self.log('Val/mr', ranks.mean())
        self.log('Val/mrr', (1.0 / ranks).mean())

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        answer = batch.pop('answer')
        retrieval_output = self.retrieval_agent.retrieve(batch=batch, device=self.device, chunk_size=int(self.args.data.eval_chunk_size), answer=answer)
        self._log_fusion_weights_4m_stats(retrieval_output.fusion_weights_4m, prefix='Test')
        tgt_rank = retrieval_output.target_ranks.detach().cpu()
        return dict(rank=tgt_rank)

    def on_test_start(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        entity_cache = self.retrieval_agent.build_entity_cache(entity_dataloader, self.device)
        self.set_entity_cache(entity_cache)

    def test_epoch_end(self, outputs):
        self.clear_entity_cache()
        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()
        self.log('Test/hits20', hits20)
        self.log('Test/hits10', hits10)
        self.log('Test/hits5', hits5)
        self.log('Test/hits3', hits3)
        self.log('Test/hits1', hits1)
        self.log('Test/mr', ranks.mean())
        self.log('Test/mrr', (1.0 / ranks).mean())

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_cfg = getattr(self.args, 'optimizer', None)
        base_lr = float(getattr(self.args, 'lr', 5e-06))
        encoder_lr = float(getattr(optimizer_cfg, 'encoder_lr', base_lr)) if optimizer_cfg else base_lr
        matcher_lr = float(getattr(optimizer_cfg, 'matcher_lr', base_lr)) if optimizer_cfg else base_lr
        fusion_lr = float(getattr(optimizer_cfg, 'fusion_lr', matcher_lr)) if optimizer_cfg else matcher_lr
        triple_lr = float(getattr(optimizer_cfg, 'triple_lr', matcher_lr)) if optimizer_cfg else matcher_lr
        weight_decay = float(getattr(optimizer_cfg, 'weight_decay', 0.0001)) if optimizer_cfg else 0.0001
        eps = float(getattr(optimizer_cfg, 'eps', 0.0001)) if optimizer_cfg else 0.0001

        def _lr_for_name(name: str) -> float:
            if name.startswith('encoder.'):
                return encoder_lr
            if any((key in name for key in ['fusion_head_4m', 'branch_calibration_4m'])):
                return fusion_lr
            if any((key in name for key in ['entity_triple_encoder', 'mention_triple_encoder', 'triple_match'])):
                return triple_lr
            return matcher_lr
        grouped = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            decay = 0.0 if any((nd in name for nd in no_decay)) else weight_decay
            group_key = (_lr_for_name(name), decay)
            grouped.setdefault(group_key, []).append(param)
        optimizer_grouped_params = [{'params': params, 'lr': lr, 'weight_decay': decay} for (lr, decay), params in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))]
        optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=base_lr, betas=(0.9, 0.999), eps=eps)
        return [optimizer]

    def _log_fusion_weights_4m_stats(self, fusion_weights_4m: torch.Tensor, prefix: str) -> None:
        if fusion_weights_4m is None:
            return
        if fusion_weights_4m.dim() != 2 or fusion_weights_4m.size(-1) != 4:
            return
        weights = fusion_weights_4m.detach()
        mean_weights = weights.mean(dim=0)
        var_weights = weights.var(dim=0, unbiased=False)
        argmax_idx = weights.argmax(dim=-1)
        batch_size = int(weights.size(0))
        self.log(f'{prefix}/fusion4m_weight_mean_text', mean_weights[0], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_mean_image', mean_weights[1], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_mean_cross', mean_weights[2], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_mean_struct', mean_weights[3], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_var_text', var_weights[0], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_var_image', var_weights[1], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_var_cross', var_weights[2], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_weight_var_struct', var_weights[3], on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_argmax_ratio_text', (argmax_idx == 0).float().mean(), on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_argmax_ratio_image', (argmax_idx == 1).float().mean(), on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_argmax_ratio_cross', (argmax_idx == 2).float().mean(), on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f'{prefix}/fusion4m_argmax_ratio_struct', (argmax_idx == 3).float().mean(), on_epoch=True, prog_bar=False, batch_size=batch_size)

    def _print_fusion_prior_4m_if_configured(self) -> None:
        model_cfg = getattr(self.args, 'model', object())
        fusion_prior_4m = getattr(model_cfg, 'fusion_prior_4m', None)
        if fusion_prior_4m is None:
            return
        try:
            prior_values = [float(item) for item in list(fusion_prior_4m)]
        except Exception:
            return
        print(f'[config] fusion_prior_4m={prior_values}')

    def _get_best_checkpoint_path(self) -> str:
        if not self.distillation_enabled:
            return ''
        callback = getattr(self.trainer, 'checkpoint_callback', None)
        if callback is not None:
            best_model_path = str(getattr(callback, 'best_model_path', '') or '')
            if best_model_path:
                return best_model_path
        logger = getattr(self, 'logger', None)
        log_dir = getattr(logger, 'log_dir', '')
        ckpt_dir = os.path.join(log_dir, 'checkpoints') if log_dir else ''
        if ckpt_dir and os.path.isdir(ckpt_dir):
            ckpt_files = sorted([os.path.join(ckpt_dir, name) for name in os.listdir(ckpt_dir) if name.endswith('.ckpt')])
            if ckpt_files:
                return ckpt_files[0]
        return ''

    def _refresh_teacher_if_needed(self) -> None:
        if not self.distillation_enabled:
            return
        if int(self.current_epoch) < int(self.teacher_start_epoch):
            return
        checkpoint_path = self._get_best_checkpoint_path()
        if not checkpoint_path:
            return
        if self.teacher_encoder is not None and self.teacher_matcher is not None:
            if self.teacher_ckpt_path == checkpoint_path:
                return
            if not self.refresh_teacher_every_epoch:
                return
        state = torch.load(checkpoint_path, map_location='cpu')
        state_dict = state['state_dict'] if 'state_dict' in state else state
        encoder_state = {key.replace('encoder.', '', 1): value for key, value in state_dict.items() if key.startswith('encoder.')}
        matcher_state = {key.replace('matcher.', '', 1): value for key, value in state_dict.items() if key.startswith('matcher.')}
        teacher_encoder = self.teacher_encoder if self.teacher_encoder is not None else PerceptionModule(self.args)
        teacher_matcher = self.teacher_matcher if self.teacher_matcher is not None else build_matching_module(self.args)
        teacher_encoder.load_state_dict(encoder_state, strict=True)
        teacher_matcher.load_state_dict(matcher_state, strict=True)
        teacher_encoder.to(self.device)
        teacher_matcher.to(self.device)
        teacher_encoder.eval()
        teacher_matcher.eval()
        for module in (teacher_encoder, teacher_matcher):
            for param in module.parameters():
                param.requires_grad = False
        self.teacher_encoder = teacher_encoder
        self.teacher_matcher = teacher_matcher
        self.teacher_ckpt_path = checkpoint_path

    def _compute_teacher_matching_output(self, mention_batch, ent_batch, ent_triple_input_ids=None, ent_triple_attention_mask=None, ent_triple_valid_mask=None):
        if not self.distillation_enabled:
            return None
        if self.teacher_encoder is None or self.teacher_matcher is None:
            return None
        with torch.no_grad():
            teacher_mention_features = self.teacher_encoder(**mention_batch)
            teacher_entity_features = self.teacher_encoder(**ent_batch)
            if hasattr(self.teacher_matcher, 'encode_entity_triples'):
                entity_triple_cls, entity_triple_tokens, entity_triple_valid_mask = self.teacher_matcher.encode_entity_triples(clip_text_model=self.teacher_encoder.clip.text_model, triple_input_ids=ent_triple_input_ids, triple_attention_mask=ent_triple_attention_mask, triple_valid_mask=ent_triple_valid_mask)
                if entity_triple_cls is not None and entity_triple_tokens is not None:
                    teacher_entity_features = teacher_entity_features.with_triple(triple_cls=entity_triple_cls, triple_tokens=entity_triple_tokens, triple_valid_mask=entity_triple_valid_mask)
            if hasattr(self.teacher_matcher, 'build_mention_triples'):
                teacher_mention_features = self.teacher_matcher.build_mention_triples(teacher_mention_features)
            return self.teacher_matcher(teacher_entity_features, teacher_mention_features)

    def compute_teacher_soft_labels(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.distillation_temperature, dim=1)

    def compute_reference_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot_label = F.one_hot(labels, num_classes=logits.size(1)).float()
        return -torch.sum(one_hot_label * F.log_softmax(logits.detach(), dim=1), dim=1, keepdim=True)

    def compute_weighted_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, alpha: float=None) -> torch.Tensor:
        if alpha is None:
            alpha = self.distillation_alpha
        teacher_soft_labels = self.compute_teacher_soft_labels(teacher_logits).detach()
        teacher_reference_loss = self.compute_reference_loss(teacher_logits, labels)
        student_reference_loss = self.compute_reference_loss(student_logits, labels)
        focal_weight = student_reference_loss / (teacher_reference_loss + 1e-07)
        focal_weight = torch.clamp(focal_weight, min=0.0)
        focal_weight = 1 - torch.exp(-focal_weight)
        soft_loss = -torch.sum(teacher_soft_labels * F.log_softmax(student_logits / self.distillation_temperature, dim=1), dim=1, keepdim=True)
        soft_loss = focal_weight * soft_loss
        soft_loss = self.distillation_temperature ** 2 * torch.mean(soft_loss)
        hard_loss = self.loss_fct(student_logits / self.distillation_temperature, labels)
        return hard_loss + alpha * soft_loss
