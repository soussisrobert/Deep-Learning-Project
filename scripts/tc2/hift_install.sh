#!/bin/bash
# HiFT on TC2, head-node side: ONLY file operations + sbatch (no python/conda/curl here). Everything stays under
# /home/msai/ning011/hift. Idempotent: verifies the uploaded archive's sha256 once, unpacks it into
# hift/repo_<commit12>, then submits the next stage's jobs that are not already queued, running or completed.
# Shared account: at most 2 submitted jobs; the first refused submission stops the loop. Never scancel.
#   bash hift_install.sh env-data     # env job, then data job (afterok dependency)
#   bash hift_install.sh step1        # B0-FT original-layout and corrected-layout runs
set -uo pipefail
cd "$HOME" || exit 1
[ "$(pwd)" = "/home/msai/ning011" ] || { echo "unexpected home $(pwd)"; exit 1; }
B=$HOME/hift; mkdir -p "$B/logs" "$B/runs"
STAGE=${1:?stage: env-data | step1}
if [ -f "$B/upload/hift_code.tar" ]; then
  (cd "$B/upload" && sha256sum -c hift_code.sha256) || { echo "archive sha256 mismatch"; exit 1; }
  C=$(cat "$B/upload/GIT_COMMIT"); R=$B/repo_${C:0:12}
  if [ -d "$R" ]; then echo "repo already unpacked at $R"; else
    mkdir "$R" && tar -xf "$B/upload/hift_code.tar" -C "$R" && cp "$B/upload/GIT_COMMIT" "$R/GIT_COMMIT" || exit 1
    (cd "$R" && sha256sum -c "$B/upload/hift_payload.sha256" >/dev/null) || { echo "payload sha256 mismatch"; exit 1; }
    echo "unpacked $R"
  fi
  rm -f "$B/upload/hift_code.tar"
fi
C=$(cat "$B/upload/GIT_COMMIT"); R=$B/repo_${C:0:12}
[ -d "$R" ] || { echo "no unpacked repo"; exit 1; }
seen() { squeue -h -u "$USER" -o "%j" | grep -qx "$1" || sacct -n -X -u "$USER" -S 2026-10-28 -o JobName%30,State | grep -qE "^ *$1 +(COMPLETED|RUNNING|PENDING)"; }
submit() {  # name sbatch-file [extra sbatch args...]
  local name=$1 file=$2; shift 2
  if seen "$name"; then echo "$name already queued/running/completed"; return 0; fi
  local out; out=$(sbatch --parsable --nodelist=TC2N03 -J "$name" --export=ALL,REPO="$R" "$@" "$R/scripts/tc2/$file" 2>&1) \
    && { echo "SUBMITTED $name=$out"; LAST=$out; return 0; } || { echo "NOT SUBMITTED $name: $(echo "$out" | head -c 120)"; return 1; }
}
echo "queue before:"; squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
case $STAGE in
  env-data)
    LAST=""; DEP=()
    if [ ! -f "$B/env/.ready" ]; then submit hift_env hift_env.sbatch || exit 0; [ -n "$LAST" ] && DEP=(--dependency=afterok:$LAST); fi
    [ -f "$B/data/.ready" ] || submit hift_data hift_data.sbatch "${DEP[@]}" ;;
  step1)
    [ -f "$B/env/.ready" ] && [ -f "$B/data/.ready" ] || { echo "env/data not ready yet"; exit 0; }
    COMMON="--backbone alexnet --init official --epochs 10 --steps-per-epoch 1000"
    submit hift_b0ft_orig hift_train.sbatch --export=ALL,REPO="$R",RUN_ID=TC2-B0FT-ORIG-001,TRAIN_ARGS="$COMMON --layout original" || exit 0
    submit hift_b0ft_corr hift_train.sbatch --export=ALL,REPO="$R",RUN_ID=TC2-B0FT-CORR-001,TRAIN_ARGS="$COMMON --layout corrected" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
