#!/bin/bash
# Run one CPU queue from plan/queue/ (no GPU). Fits next to a GPU queue job.
#   sbatch --export=ALL,QUEUE=plan/queue/03_cpu_P2.tsv slurm/job_queue_cpu_tc2.sbatch
#SBATCH --job-name=hift_qcpu
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=runs/queue_cpu_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env_tc2.sh
echo "queue: ${QUEUE:?sbatch --export=ALL,QUEUE=plan/queue/<file>}"
bash scripts/run_queue.sh "$QUEUE"
