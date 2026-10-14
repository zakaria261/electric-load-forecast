#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_data.py

printf 'Environnement prêt. Lancez : .venv/bin/python scripts/run_rolling_cv.py --models baselines\n'
