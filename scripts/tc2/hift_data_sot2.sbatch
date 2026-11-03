#!/bin/bash
# HiFT TC2 stage 5: VisDrone2019-SOT train_part2 (43 sequences, 12.62 GiB) into ~/hift/data, by HTTP range reads of
# the same Hugging Face archive as part1 (huseyincavus/visdrone2019-sot; per-member CRC verified, Hub LFS oid
# recorded). Then an exact-duplicate frame check of part2 against V1 (SOT-val) and V2 (MOT-val); the part2 frame
# hash list is kept for a local check against the sealed SOT test-dev. Owner approval 2026-11-03 (stage 5).
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/msai/ning011/hift/logs/%x_%j.out
#SBATCH --error=/home/msai/ning011/hift/logs/%x_%j.out
set -eo pipefail   # no -u: the cluster's conda activate.d scripts reference unset variables
B=$HOME/hift; R=${REPO:?REPO=path of the unpacked code}
date --iso-8601=seconds; hostname
[ -f "$B/env/.ready" ] || { echo "env not ready"; exit 9; }
set +e   # cluster conda hooks may return non-zero internally; checked explicitly below
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate "$B/env"
set -e
[ "${CONDA_PREFIX:-}" = "$B/env" ] || { echo "env activation failed"; exit 4; }
export PYTHONPATH=$R/src:$R/scripts PYTHONUNBUFFERED=1
D=$B/data; L=$B/ledgers
SOT=$D/visdrone/SOT; P2=$SOT/VisDrone2019-SOT-train_part2/VisDrone2019-SOT-train
if [ -f "$L/SOT2.json" ]; then echo "skip SOT2.json (already acquired)"; else
  python -u "$R/scripts/acquire_hf_zip_members.py" --repo huseyincavus/visdrone2019-sot --root "$SOT" \
    --ledger "$L/SOT2.json" --files visdrone2019-sot.zip --include VisDrone2019-SOT-train_part2/
fi
n_seq=$(ls "$P2/sequences" | wc -l); n_ann=$(ls "$P2/annotations" | wc -l)
echo "part2: $n_seq sequences, $n_ann annotation files"
[ "$n_seq" -eq 43 ] && [ "$n_ann" -eq 43 ] || { echo "part2 layout check FAILED"; exit 11; }
python -u "$R/scripts/frame_overlap.py" --train "$P2/sequences" \
  --eval V1="$SOT/VisDrone2019-SOT-val/VisDrone2019-SOT-val/sequences" V2="$D/visdrone/MOT/VisDrone2019-MOT-val/sequences" \
  --out "$L/SOT2-overlap.json" --hashes-out "$L/SOT2-frames.sha256"
du -sh "$SOT"/*
echo "SOT2 READY $(date --iso-8601=seconds)"
