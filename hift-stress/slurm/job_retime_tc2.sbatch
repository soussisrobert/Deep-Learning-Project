#!/bin/bash
# Matched re-timing of fp32 vs the accuracy-tested int8bb-s1 model (CPU only). See AGENT_RUNBOOK_RETIME.md.
#   sbatch --export=ALL,THREADS=4,SESSIONS=6 slurm/job_retime_tc2.sbatch     (about 1.7 h)
#   sbatch --export=ALL,THREADS=1,SESSIONS=4 slurm/job_retime_tc2.sbatch     (about 1.9 h; submit after the first ends)
#SBATCH --job-name=hift_retime
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=runs/retime_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
T=${THREADS:?THREADS=4 or 1}; S=${SESSIONS:-6}
python scripts/retime_matched.py --snapshot weights/general_model.pth --dataset flat --root data/uav123_flat \
  --ref-int8 results/uav123-sub40_int8sel_clean_nodef_t4_cal1 --ref-fp32 results/uav123_fp32_clean_nodef_t4 \
  --threads "$T" --sessions "$S" --out "results/retime_matched_t$T"
