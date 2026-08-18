#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <config.yaml> <checkpoint.ckpt>"
  exit 1
fi
python main.py --config "$1" --test-only-ckpt "$2"
