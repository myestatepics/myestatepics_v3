#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
echo "Installed. Production command:"
echo ".venv/bin/python tools/process_dng_batch.py --input input --output output/v5_production"
