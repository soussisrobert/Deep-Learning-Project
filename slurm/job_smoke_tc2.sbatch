#!/bin/bash
# GPU check + timing probe (~15 min). Run this before any long job:
#   sbatch slurm/job_smoke_tc2.sbatch
# It runs the pipeline checks with the real weights, then a small clean / PGD run on
# 5 benchmark sequences so you can estimate how long the full grids take.
#SBATCH --job-name=hift_smoke
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=00:45:00
#SBATCH --output=runs/smoke_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
python scripts/smoke_test.py --snapshot weights/general_model.pth --device cuda
args=(--dataset "$DATASET" --root "$ROOT" --snapshot weights/general_model.pth --device cuda --max-seqs 5 --out results_smoke)
python scripts/run_track.py "${args[@]}"
python scripts/run_track.py "${args[@]}" --attack pgd --eps 4 --steps 10
python - <<'EOF'
import json, glob, os, sys
sys.path.insert(0, ".")
from hiftstress.data import load_dataset
n = sum(len(s) for s in load_dataset(os.environ["DATASET"], os.environ["ROOT"]))
print(f"benchmark {os.environ['ROOT']}: {n} frames")
for f in sorted(glob.glob("results_smoke/*/summary.json")):
    s = json.load(open(f))
    per_frame = s["ms_per_frame"]["total"]
    print(f"{s['run']}: AUC {s['success_auc']:.3f}  {per_frame:.1f} ms/frame  "
          f"-> full benchmark ~ {per_frame * n / 60000:.0f} min per run")
EOF
