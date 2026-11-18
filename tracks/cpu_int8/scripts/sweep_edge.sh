#!/bin/bash
# Edge-cost grid: cost benchmark per variant + clean tracking accuracy per variant.
#
#   PART=gpu  : fp32 / fp16 / prune30-70 / torch.compile on the GPU
#   PART=cpu  : fp32 / int8dyn / int8bb / prune50 on CPU at 1, 2, 4 threads
#               (a stand-in for a drone's onboard CPU; name the CPU model in the report)
#
# Same environment variables as sweep_adv.sh. Finished tracking runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PART=${PART:?set PART=gpu or cpu}
DATASET=${DATASET:-dtb70}; ROOT=${ROOT:-data/DTB70}; SNAP=${SNAP:-weights/general_model.pth}
OUT=${OUT:-results}; SEED=${SEED:-0}; ITERS=${ITERS:-200}
MAXSEQ=${MAXSEQ-5}; MAXFRAMES=${MAXFRAMES-60}
PY=${PY:-python}
TAG=${TAG:-$DATASET}   # prefix of run names, e.g. TAG=uav123

if [[ "$SNAP" == random ]]; then W=(--allow-random); else W=(--snapshot "$SNAP"); fi  # SNAP=random: pipeline test
common=(--dataset "$DATASET" --root "$ROOT" "${W[@]}" --out "$OUT" --seed "$SEED")
[[ -n "$MAXSEQ" ]] && common+=(--max-seqs "$MAXSEQ")
[[ -n "$MAXFRAMES" ]] && common+=(--max-frames "$MAXFRAMES")
sub=""; [[ -n "$MAXSEQ" ]] && sub="-sub$MAXSEQ"

track() {  # track <variant> <device> [threads]
  local name="${TAG}${sub}_$1_clean_nodef${3:+_t$3}"
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "[skip] $name (done)"; return; fi
  echo "[$(date +%H:%M:%S)] $name"
  $PY scripts/run_track.py "${common[@]}" --name "$name" --variant "$1" --device "$2" ${3:+--threads $3}
}

if [[ "$PART" == gpu ]]; then
  $PY scripts/bench_edge.py "${W[@]}" --dataset "$DATASET" --root "$ROOT" --device cuda \
      --variants fp32 fp16 prune30 prune50 prune70 compile --iters "$ITERS" --out "$OUT/edge"
  for v in fp32 fp16 prune30 prune50 prune70; do track "$v" cuda; done
elif [[ "$PART" == cpu ]]; then
  $PY scripts/bench_edge.py "${W[@]}" --dataset "$DATASET" --root "$ROOT" --device cpu \
      --variants fp32 int8dyn int8bb prune50 --threads ${THREADS:-1 2 4} --iters "$ITERS" --out "$OUT/edge"
  for v in fp32 int8dyn int8bb; do track "$v" cpu "${TRACK_THREADS:-4}"; done
  # cross-track: does the int8 model inherit the fp32 model's weakness? (attack computed on fp32)
  name="${TAG}${sub}_int8bb_pgd4s10_nodef"
  [[ -f "$OUT/$name/summary.json" ]] || $PY scripts/run_track.py "${common[@]}" --name "$name" \
      --variant int8bb --device cpu --threads "${TRACK_THREADS:-4}" --attack pgd --eps 4 --steps 10
else
  echo "unknown PART $PART"; exit 1
fi
echo "edge part $PART finished"
