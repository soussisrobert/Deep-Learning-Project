#!/bin/bash
#SBATCH --job-name=hift_cpu_probe
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/cpu_probe_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
args=(--dataset "$DATASET" --root "$ROOT" --snapshot weights/general_model.pth
      --device cpu --threads 4 --max-seqs 1 --max-frames 30 --out results_probe)
python scripts/run_track.py "${args[@]}" --name cpu_fp32_clean_probe
python scripts/run_track.py "${args[@]}" --variant int8bb --name cpu_int8bb_clean_probe
python scripts/run_track.py "${args[@]}" --variant int8bb --attack pgd --eps 4 --steps 10 --name cpu_int8bb_pgd_probe
