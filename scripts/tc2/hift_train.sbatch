#!/bin/bash
# One HiFT domain training run on TC2N03 (L40S). RUN_ID and TRAIN_ARGS come from the installer (--export).
# Data = the TC2 aerial subset in ~/hift/data; runs under ~/hift/runs/<RUN_ID>. Never resumes: an existing run dir aborts.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --output=/home/msai/ning011/hift/logs/%x_%j.out
#SBATCH --error=/home/msai/ning011/hift/logs/%x_%j.out
set -o pipefail   # no -u: the cluster's conda activate.d scripts reference unset variables
B=$HOME/hift; R=${REPO:?REPO}; RUN_ID=${RUN_ID:?RUN_ID}; TRAIN_ARGS=${TRAIN_ARGS:?TRAIN_ARGS}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$B/env/.ready" ] || { echo "env not ready"; exit 9; }
[ -f "$B/data/.ready" ] || { echo "data not ready"; exit 10; }
[ -e "$B/runs/$RUN_ID" ] && { echo "run dir exists: never resume; move it aside by hand"; exit 16; }
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate "$B/env"
[ "${CONDA_PREFIX:-}" = "$B/env" ] || { echo "env activation failed"; exit 4; }
export PYTHONPATH=$R/src PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$R"
echo "=== $RUN_ID (git $(cat GIT_COMMIT)) args: $TRAIN_ARGS"
# shellcheck disable=SC2086
python -u scripts/train_b2_domain.py --run-id "$RUN_ID" --output-root "$B/runs" --data-root "$B/data" \
  --train-parts visdrone-sot-1 visdrone-mot uavdt \
  --source-weights '{"VisDrone-SOT": 0.4, "VisDrone-MOT": 0.35, "UAVDT": 0.25}' \
  --workers 9 --owner-instruction "Owner chat 2026-10-28: run HiFT domain training on TC2 (steps 1 then 3)." $TRAIN_ARGS
STATUS=$?
echo "=== exit $STATUS $(date --iso-8601=seconds)"
exit $STATUS
