#!/bin/bash
# CPU-only stages keep the long CPU attack out of clean evaluation jobs.
#SBATCH --job-name=hift_cpu_edge
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=runs/cpu_edge_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
stage=${CPU_STAGE:-bench_clean}
variant=${CPU_VARIANT:-fp32}
out=${OUT:-results}
args=(--dataset "$DATASET" --root "$ROOT" --snapshot weights/general_model.pth)
case "$stage" in bench|clean|bench_clean) ;; *) echo "Unknown CPU_STAGE: $stage"; exit 1 ;; esac
if [[ "$stage" == bench || "$stage" == bench_clean ]]; then
  python scripts/bench_edge.py "${args[@]}" --device cpu \
    --variants fp32 int8dyn int8bb prune50 --threads 1 2 4 --iters 200 --out "$out/edge"
fi
if [[ "$stage" == clean || "$stage" == bench_clean ]]; then
  name="${TAG}_${variant}_clean_nodef_t4"
  if [[ ! -f "$out/$name/summary.json" ]]; then
    python scripts/run_track.py "${args[@]}" --device cpu --threads 4 \
      --variant "$variant" --seed 0 --name "$name" --out "$out"
  else
    echo "[skip] $name (done)"
  fi
fi
python scripts/summarize.py --results "$out"
