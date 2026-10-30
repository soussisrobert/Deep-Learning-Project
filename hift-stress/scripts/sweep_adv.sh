#!/bin/bash
# Adversarial-robustness grid. Every run is one call to run_track.py; finished runs are skipped,
# so a job that hits the wall time can simply be resubmitted.
#
#   PHASE=A  attack strength : clean, noise / FGSM / PGD-10 at several budgets (fp32)
#   PHASE=B  defences        : 4 input transforms x {clean, PGD, adaptive PGD} at EPS_DEF
#   PHASE=C  cross-track     : the same PGD perturbation (computed on fp32) against compressed models
#
# Settings come from environment variables (defaults = quick local run):
#   DATASET=dtb70  ROOT=data/DTB70  SNAP=weights/general_model.pth  DEVICE=auto
#   MAXSEQ=5  MAXFRAMES=60        (leave both empty for the full benchmark)
#   OUT=results  EPS_DEF=4  STEPS=10  SEED=0
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE=${PHASE:?set PHASE=A, B or C}
DATASET=${DATASET:-dtb70}; ROOT=${ROOT:-data/DTB70}; SNAP=${SNAP:-weights/general_model.pth}
DEVICE=${DEVICE:-auto}; OUT=${OUT:-results}; EPS_DEF=${EPS_DEF:-4}; STEPS=${STEPS:-10}; SEED=${SEED:-0}
MAXSEQ=${MAXSEQ-5}; MAXFRAMES=${MAXFRAMES-60}
PY=${PY:-python}
TAG=${TAG:-$DATASET}   # prefix of run names, e.g. TAG=uav123

if [[ "$SNAP" == random ]]; then W=(--allow-random); else W=(--snapshot "$SNAP"); fi  # SNAP=random: pipeline test
common=(--dataset "$DATASET" --root "$ROOT" "${W[@]}" --out "$OUT" --seed "$SEED" --steps "$STEPS")
[[ -n "$MAXSEQ" ]] && common+=(--max-seqs "$MAXSEQ")
[[ -n "$MAXFRAMES" ]] && common+=(--max-frames "$MAXFRAMES")
sub=""; [[ -n "$MAXSEQ" ]] && sub="-sub$MAXSEQ"

run() {  # run <expected run-name suffix> <args...>
  # Optional exact suffix list for batches that fit the six-hour wall limit.
  if [[ -n ${RUN_ONLY:-} && " $RUN_ONLY " != *" $1 "* ]]; then return; fi
  local name="${TAG}${sub}_$1"; shift
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "[skip] $name (done)"; return; fi
  echo "[$(date +%H:%M:%S)] $name"
  $PY scripts/run_track.py "${common[@]}" --name "$name" "$@"
}

case "$PHASE" in
A)
  run fp32_clean_nodef --device "$DEVICE"
  for e in 2 4 8; do run "fp32_noise${e}_nodef" --device "$DEVICE" --attack noise --eps "$e"; done
  for e in 1 2 4 8; do run "fp32_fgsm${e}_nodef" --device "$DEVICE" --attack fgsm --eps "$e"; done
  for e in 1 2 4 8; do run "fp32_pgd${e}s${STEPS}_nodef" --device "$DEVICE" --attack pgd --eps "$e"; done
  ;;
B)
  for d in blur1 median3 bits5 jpeg75; do
    run "fp32_clean_${d}" --device "$DEVICE" --defense "$d"
    run "fp32_pgd${EPS_DEF}s${STEPS}_${d}" --device "$DEVICE" --attack pgd --eps "$EPS_DEF" --defense "$d"
    run "fp32_pgd${EPS_DEF}s${STEPS}_${d}_adaptive" --device "$DEVICE" --attack pgd --eps "$EPS_DEF" --defense "$d" --adaptive
  done
  ;;
C)
  # GPU variants here; the int8 (CPU) variants run in the CPU edge job.
  for v in ${XVARIANTS:-fp16 prune30 prune50}; do
    run "${v}_pgd${EPS_DEF}s${STEPS}_nodef" --device "$DEVICE" --variant "$v" --attack pgd --eps "$EPS_DEF"
  done
  ;;
*) echo "unknown PHASE $PHASE"; exit 1 ;;
esac
echo "phase $PHASE finished"
