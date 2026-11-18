#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --job-name=HiFT_night_run
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Night fine-tuning run. usage: sbatch job_night_run.sh <A|B|C> [resume checkpoint]
# 10 epochs of 330k pairs (about 16 min each at 350 pairs/s) fit in one job; a checkpoint and the val losses are written after every epoch,
# --time_budget_s 20400 (5 h 40 min) stops after an epoch if the next one might not fit, then rerun with the last checkpoint as 2nd argument.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
RUN=$1
RESUME=${2:-}
EXTRA=""
if [ -n "$RESUME" ]; then EXTRA="--resume $RESUME"; fi
python -I tools/train_night.py --cfg experiments/night_ft/config.yaml experiments/night_ft/run_$RUN.yaml --run $RUN --time_budget_s 20400 $EXTRA
