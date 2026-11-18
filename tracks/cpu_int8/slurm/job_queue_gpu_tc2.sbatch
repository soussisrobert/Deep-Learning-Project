#!/bin/bash
# Run one GPU (or GPU + CPU) queue from plan/queue/. Sized to leave 4 CPUs / 10 GB for a CPU job
# alongside it (TC2 normal QoS: 2 jobs, 10 CPUs, 30 GB, 1 GPU in total).
#   sbatch --export=ALL,QUEUE=plan/queue/01_gpu_P1.tsv slurm/job_queue_gpu_tc2.sbatch
#SBATCH --job-name=hift_qgpu
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=runs/queue_gpu_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
echo "queue: ${QUEUE:?sbatch --export=ALL,QUEUE=plan/queue/<file>}"
bash scripts/run_queue.sh "$QUEUE"
