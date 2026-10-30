#!/bin/bash
# Validation-only job (no training): tracks existing checkpoints on validation V1 (VisDrone-SOT-val) + V2
# (VisDrone-MOT-val episodes) into ~/hift/validation/<label>. V3 is omitted: the existing runs trained on all of UAVDT.
# VAL_JOBS = ";"-separated "label|validate_domain.py args" (from the installer). Test sets are never read.
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
B=$HOME/hift; R=${REPO:?REPO}; VAL_JOBS=${VAL_JOBS:?VAL_JOBS}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$B/env/.ready" ] && [ -f "$B/data/.ready" ] || { echo "env/data not ready"; exit 9; }
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate "$B/env"
[ "${CONDA_PREFIX:-}" = "$B/env" ] || { echo "env activation failed"; exit 4; }
export PYTHONPATH=$R/src PYTHONUNBUFFERED=1
cd "$R"; mkdir -p "$B/validation"; STATUS=0
IFS=';' read -ra ITEMS <<< "$VAL_JOBS"
for item in "${ITEMS[@]}"; do
  label=${item%%|*}; vargs=${item#*|}
  out=$B/validation/$label
  [ -f "$out/validation.json" ] && { echo "skip $label (done)"; continue; }
  [ -e "$out" ] && out=$out-retry$(date +%s)   # a failed attempt's directory is kept, never reused
  echo "=== $label -> $out: $vargs $(date --iso-8601=seconds)"
  # shellcheck disable=SC2086
  python -u scripts/validate_domain.py $vargs --data-root "$B/data" --output "$out" --subsets V1 V2     | grep VALIDATION || STATUS=1
done
echo "=== exit $STATUS $(date --iso-8601=seconds)"
exit $STATUS
