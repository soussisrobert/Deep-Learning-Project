#!/bin/bash
# One-time setup on your laptop (macOS or Linux). Run from the hift-stress folder:
#   bash setup_local.sh
# Creates .venv, downloads the official HiFT code at a fixed commit, makes it run
# without CUDA, then runs the pipeline checks (no dataset or weights needed yet).
set -euo pipefail
cd "$(dirname "$0")"
PIN=992aa7c1605360d607af84cb804b93cd8ee81caf

if [[ ! -x .venv/bin/python ]]; then
  # PyTorch wheels lag behind the newest Python; prefer 3.12 / 3.11 when installed
  PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3)
  echo "using $PYBIN ($($PYBIN --version))"
  "$PYBIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt

if [[ ! -d third_party/HiFT/pysot ]]; then
  mkdir -p third_party
  git clone https://github.com/vision4robotics/HiFT.git third_party/HiFT
  git -C third_party/HiFT checkout "$PIN"
fi
python tools/patch_hift.py third_party/HiFT
mkdir -p weights data results
python scripts/smoke_test.py
echo
echo "Setup done. Next: bash get_assets.sh  (weights + DTB70), then the quick runs in README.md."
