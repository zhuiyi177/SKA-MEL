import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from codes.backbone.lightning_skamel import SKAMELModel
from codes.utils.dataset import DataModuleForSKAMEL
from codes.utils.functions import setup_parser


def main():
    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModuleForSKAMEL(args)
    run_root = os.path.join(os.path.abspath(args.project_root), 'runs')
    logger = pl.loggers.CSVLogger(run_root, name=args.run_name, flush_logs_every_n_steps=30)

    trainer_cfg = OmegaConf.to_container(args.trainer, resolve=True) or {}
    early_stopping_enabled = bool(trainer_cfg.pop('enable_early_stopping', True))
    monitor = str(trainer_cfg.pop('early_stopping_monitor', 'Val/mrr'))
    mode = str(trainer_cfg.pop('early_stopping_mode', 'max'))
    patience = int(trainer_cfg.pop('early_stopping_patience', 3))
    test_only_ckpt = str(getattr(args, 'test_only_ckpt', '') or '')

    checkpointing_enabled = bool(trainer_cfg.get('enable_checkpointing', True))
    if test_only_ckpt:
        trainer_cfg['enable_checkpointing'] = False

    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor=monitor, save_weights_only=True, mode=mode)
    if early_stopping_enabled and not test_only_ckpt:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=True))
    if checkpointing_enabled and not test_only_ckpt:
        callbacks.append(ckpt_callback)

    trainer = pl.Trainer(
        **trainer_cfg,
        deterministic=True,
        logger=logger,
        default_root_dir=run_root,
        callbacks=callbacks,
    )

    if test_only_ckpt:
        if not os.path.exists(test_only_ckpt):
            raise FileNotFoundError(f'checkpoint not found: {test_only_ckpt}')
        trainer.test(SKAMELModel(args), datamodule=data_module, ckpt_path=test_only_ckpt)
        return

    model = SKAMELModel(args)
    trainer.fit(model, datamodule=data_module)
    best_path = str(getattr(ckpt_callback, 'best_model_path', '') or '') if checkpointing_enabled else ''
    if best_path:
        trainer.test(SKAMELModel(args), datamodule=data_module, ckpt_path=best_path)
    else:
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
