#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root using Git Bash on Windows.
python -m venv .venv
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
# Disable pip HTTP cache: local Windows cache tempfile creation was observed looping.
"$PY" -m pip install --no-cache-dir torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir -r requirements/lock-windows-py313-cu128.txt --extra-index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir --no-build-isolation -e .
"$PY" -m pip freeze > artifacts/environment.freeze.txt
"$PY" scripts/diagnose_environment.py --output artifacts/environment.json
"$PY" -m pytest
