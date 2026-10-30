#!/bin/bash
# Adversarial grid on the full benchmark (UAV123 by default), one phase per job (each fits the 6 h wall; re-submit to resume):
#   sbatch --export=ALL,PHASE=A slurm/job_adv_tc2.sbatch     # attack strength
#   sbatch --export=ALL,PHASE=B slurm/job_adv_tc2.sbatch     # defences (+ adaptive attacks)
#   sbatch --export=ALL,PHASE=C slurm/job_adv_tc2.sbatch     # PGD vs fp16 / pruned models
# Submit from the login node; do not chain sbatch inside a job (TC2 refuses it).
#SBATCH --job-name=hift_adv
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=runs/adv_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
export PHASE=${PHASE:?sbatch --export=ALL,PHASE=A}
export SNAP=weights/general_model.pth   # DATASET / ROOT / TAG come from env_tc2.sh
export DEVICE=cuda OUT=${OUT:-results} MAXSEQ=${MAXSEQ-} MAXFRAMES=${MAXFRAMES-}
bash scripts/sweep_adv.sh
python scripts/summarize.py --results "$OUT"
