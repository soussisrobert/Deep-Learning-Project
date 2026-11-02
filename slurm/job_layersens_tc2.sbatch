#!/bin/bash
# Per-layer int8 sensitivity (CPU only). See AGENT_RUNBOOK_LAYERSENS.md.
#   sbatch --export=ALL,STAGE=fidelity slurm/job_layersens_tc2.sbatch                 (~45 min, run alone)
#   sbatch --export=ALL,STAGE=track,DRAWS="1" slurm/job_layersens_tc2.sbatch          (~1.5 h)
#   sbatch --export=ALL,STAGE=track,DRAWS="2 3" slurm/job_layersens_tc2.sbatch        (~3 h)
#SBATCH --job-name=hift_layersens
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=runs/layersens_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
STAGE=${STAGE:?STAGE=fidelity or track}
python scripts/layer_sensitivity.py --stage "$STAGE" --snapshot weights/general_model.pth --root data/uav123_flat \
  --results results --out results/layer_sensitivity --draws ${DRAWS:-1 2 3}
python scripts/layer_sensitivity.py --stage report --results results --out results/layer_sensitivity --draws 1 2 3 || true
