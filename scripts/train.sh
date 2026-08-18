#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/wikimel.yaml}
python main.py --config "$CONFIG"
