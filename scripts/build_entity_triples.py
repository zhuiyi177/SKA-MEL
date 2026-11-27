#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf
from transformers import CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.utils.functions import resolve_project_path
from codes.utils.triple_builder import build_dataset_entity_triples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None)
    p.add_argument('--entity', default=None)
    p.add_argument('--output-json', default=None)
    p.add_argument('--output-cache', default=None)
    p.add_argument('--pretrained-model', default=None)
    p.add_argument('--triple-max-num', type=int, default=None)
    p.add_argument('--triple-text-max-length', type=int, default=None)
    a = p.parse_args()

    cfg = OmegaConf.load(resolve_project_path(str(ROOT), a.config)) if a.config else None

    entity = a.entity or (str(cfg.data.entity) if cfg is not None else None)
    output_json = a.output_json or (str(cfg.data.entity_triple_file) if cfg is not None else None)
    output_cache = a.output_cache or (str(cfg.data.entity_triple_token_cache) if cfg is not None else None)
    pretrained_model = a.pretrained_model or (
        str(cfg.pretrained_model) if cfg is not None else 'checkpoint/clip-vit-base-patch32'
    )
    triple_max_num = (
        a.triple_max_num
        if a.triple_max_num is not None
        else (int(cfg.data.triple_max_num) if cfg is not None else 3)
    )
    triple_text_max_length = (
        a.triple_text_max_length
        if a.triple_text_max_length is not None
        else (int(cfg.data.triple_text_max_length) if cfg is not None else 16)
    )

    if not (entity and output_json and output_cache):
        p.error('Provide --config <yaml> or explicitly pass --entity, --output-json, and --output-cache.')

    processor = CLIPProcessor.from_pretrained(resolve_project_path(str(ROOT), pretrained_model))
    build_dataset_entity_triples(
        kb_entity_path=resolve_project_path(str(ROOT), entity),
        output_json_path=resolve_project_path(str(ROOT), output_json),
        output_token_cache_path=resolve_project_path(str(ROOT), output_cache),
        tokenizer=processor.tokenizer,
        triple_max_num=triple_max_num,
        triple_text_max_length=triple_text_max_length,
    )


if __name__ == '__main__':
    main()
