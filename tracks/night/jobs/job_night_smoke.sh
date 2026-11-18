#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_night_smoke
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Phase 2 smoke test: 200 iterations of run C (all options on, the heaviest data path), batch 220, from first.pth.
# Reports loss, GPU memory, speed, time per epoch. No checkpoint is written. 5 validation batches check the validation path.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
nvidia-smi --query-gpu=name,memory.total --format=csv
python -I tools/train_night.py --cfg experiments/night_ft/config.yaml experiments/night_ft/run_C.yaml --run smoke --max_iters 200 --val_batches 5
