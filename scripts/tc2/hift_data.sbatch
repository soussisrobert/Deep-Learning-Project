#!/bin/bash
# HiFT TC2 training data (~24.5 GB) into ~/hift/data, downloaded INSIDE a job via HTTP range reads of the
# same Hugging Face archives used locally (only the needed members; per-member CRC verified):
#   VisDrone2019-MOT train+val (vanthanh/VisDrone2019-MOT), VisDrone2019-SOT train_part1 + val
#   (huseyincavus/visdrone2019-sot), UAVDT benchmark-M + MOTD GT (vanthanh/UAVDT-Benchmark-M).
# LaSOT and VisDrone-SOT train_part2 are deliberately omitted (home quota). Test sets are never downloaded.
# Each part is skipped when its ledger already exists. Writes ~/hift/data/.ready when all parts are present.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/msai/ning011/hift/logs/%x_%j.out
#SBATCH --error=/home/msai/ning011/hift/logs/%x_%j.out
set -euo pipefail
B=$HOME/hift; R=${REPO:?REPO=path of the unpacked code}
date --iso-8601=seconds; hostname
[ -f "$B/env/.ready" ] || { echo "env not ready"; exit 9; }
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate "$B/env"
export PYTHONPATH=$R/src:$R/scripts PYTHONUNBUFFERED=1
D=$B/data; L=$B/ledgers; mkdir -p "$D" "$L"
get() {  # ledger repo root files... [-- include tokens...]
  local ledger=$1 repo=$2 root=$3; shift 3
  if [ -f "$L/$ledger" ]; then echo "skip $ledger (already acquired)"; return 0; fi
  python -u "$R/scripts/acquire_hf_zip_members.py" --repo "$repo" --root "$root" --ledger "$L/$ledger" "$@"
}
get MOT.json vanthanh/VisDrone2019-MOT "$D/visdrone/MOT" --files VisDrone2019-MOT-val.zip VisDrone2019-MOT-train.zip
get SOT.json huseyincavus/visdrone2019-sot "$D/visdrone/SOT" --files visdrone2019-sot.zip \
    --include VisDrone2019-SOT-train_part1/ VisDrone2019-SOT-val/
get UAVDT.json vanthanh/UAVDT-Benchmark-M "$D/uavdt" --files UAV-benchmark-MOTD_v1.0.zip M_attr.zip UAV-benchmark-M.zip
touch "$D/.ready"
du -sh "$D"
echo "DATA READY $(date --iso-8601=seconds)"
