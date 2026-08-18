#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from transformers import CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.utils.functions import resolve_project_path
from codes.utils.triple_builder import build_dataset_entity_triples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--entity', required=True)
    p.add_argument('--output-json', required=True)
    p.add_argument('--output-cache', required=True)
    p.add_argument('--pretrained-model', default='checkpoint/clip-vit-base-patch32')
    p.add_argument('--triple-max-num', type=int, default=3)
    p.add_argument('--triple-text-max-length', type=int, default=16)
    a = p.parse_args()

    processor = CLIPProcessor.from_pretrained(resolve_project_path(str(ROOT), a.pretrained_model))
    build_dataset_entity_triples(
        kb_entity_path=resolve_project_path(str(ROOT), a.entity),
        output_json_path=resolve_project_path(str(ROOT), a.output_json),
        output_token_cache_path=resolve_project_path(str(ROOT), a.output_cache),
        tokenizer=processor.tokenizer,
        triple_max_num=a.triple_max_num,
        triple_text_max_length=a.triple_text_max_length,
    )


if __name__ == '__main__':
    main()
