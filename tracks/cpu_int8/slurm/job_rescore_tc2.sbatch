#!/bin/bash
# Step 0 (minutes, CPU only): re-score every finished run with both precision rules and per
# UAV123 attribute, plus the int8full cost benchmark that the next round compares against.
#   sbatch slurm/job_rescore_tc2.sbatch
#SBATCH --job-name=hift_rescore
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=runs/rescore_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
# Attribute labels: pysot UAV.json if the archive has one, else KAUST anno/UAV123/att/*.txt
UAVJSON=$(find -L data -maxdepth 4 -name UAV.json 2>/dev/null | head -1 || true)
ATTDIR=$(find -L data -maxdepth 5 -type d -path "*anno/UAV123/att" 2>/dev/null | head -1 || true)
ATTR=(); if [[ -n "$UAVJSON" ]]; then ATTR=(--uav-json "$UAVJSON"); elif [[ -n "$ATTDIR" ]]; then ATTR=(--att-dir "$ATTDIR"); fi
echo "attribute source: ${ATTR[*]:-none (overall scores only)}"
python scripts/rescore.py --dataset "$DATASET" --root "$ROOT" --results results "${ATTR[@]}" --out results/rescore
python scripts/bench_edge.py --snapshot weights/general_model.pth --dataset "$DATASET" --root "$ROOT" \
  --device cpu --variants fp32 int8bb int8full --threads 1 2 4 --iters 200 --out results/edge_round2   # keeps the verified round-1 rows untouched
python scripts/summarize.py --results results
