#!/bin/bash
# Edge-cost grid on the full benchmark (UAV123 by default):
#   sbatch --export=ALL,PART=gpu slurm/job_edge_tc2.sbatch   # fp32 / fp16 / pruned / compiled on the GPU
#   sbatch --export=ALL,PART=cpu slurm/job_edge_tc2.sbatch   # fp32 / int8 on 1-4 CPU threads
# The CPU part still sits on the GPU partition (that is where our QoS is); it does not use the GPU.
#SBATCH --job-name=hift_edge
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=runs/edge_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
export PART=${PART:?sbatch --export=ALL,PART=gpu}
export SNAP=weights/general_model.pth   # DATASET / ROOT / TAG come from env_tc2.sh
export OUT=${OUT:-results} MAXSEQ=${MAXSEQ-} MAXFRAMES=${MAXFRAMES-}
bash scripts/sweep_edge.sh
python scripts/summarize.py --results "$OUT"
