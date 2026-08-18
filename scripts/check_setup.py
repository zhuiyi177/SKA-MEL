#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/wikimel.yaml')
    p.add_argument('--allow-missing-data', action='store_true')
    a = p.parse_args()
    cfg = OmegaConf.load(ROOT / a.config)

    required = ['qid2id', 'entity', 'train_file', 'dev_file', 'test_file', 'kb_img_folder', 'mention_img_folder']
    missing = []
    for key in required:
        path = ROOT / str(cfg.data[key])
        if not path.exists():
            missing.append(str(path))
    print(f'config: {a.config}')
    print(f'entities: {cfg.data.num_entity}')
    print(f'base weights: {list(cfg.model.fusion_prior_4m)}')
    if missing:
        print('missing local data paths:')
        for path in missing:
            print(f'  - {path}')
        if not a.allow_missing_data:
            raise SystemExit(1)
    print('setup check passed')


if __name__ == '__main__':
    main()
