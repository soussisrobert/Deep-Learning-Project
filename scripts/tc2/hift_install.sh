#!/bin/bash
# HiFT on TC2, head-node side: ONLY file operations + sbatch (no python/conda/curl here). Everything stays under
# /home/msai/ning011/hift. Idempotent: verifies the uploaded archive's sha256 once, unpacks it into
# hift/repo_<commit12>, then submits the next stage's jobs that are not already queued, running or completed.
# Shared account: at most 2 submitted jobs; the first refused submission stops the loop. Never scancel.
#   bash hift_install.sh env-data     # env job, then data job (afterok dependency)
#   bash hift_install.sh step1        # B0-FT original-layout and corrected-layout runs
#   bash hift_install.sh submit NAME [ATTEMPT]  # one job from the table below (used by the local orchestrator);
#                                     ATTEMPT 2.. reruns a failed job under a NEW run id (-002, ...), never reusing a run dir
#   bash hift_install.sh unpack       # verify + unpack an uploaded archive only
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
  unpack) ;;
  submit)
    NAME=${2:?job name}; JOB=$NAME; SEEDK=""
    case $NAME in *_s[2-9]) SEEDK=${NAME##*_s}; NAME=${NAME%_s[2-9]} ;; esac   # seed replicate k of a table job
    [ -f "$B/env/.ready" ] && [ -f "$B/data/.ready" ] || { echo "NOT SUBMITTED $NAME: env/data not ready"; exit 0; }
    if [ "$NAME" = hift_val_diag ] || [ "$NAME" = hift_val_diag2 ]; then  # stage-2 diagnostics D1/D2 (diag2: rerun after the git-identity fix): validation only, existing checkpoints (V1+V2)
      V="B0-RAW-UPSTREAM|--official --score raw --selection upstream;B0-SIGMOID-NANSAFE|--official"
      for r in TC2-B0FT-CORR-001 TC2-B0FT-CORR-S2-001 TC2-B0FT-CORR-S3-001 TC2-B0FT-ORIG-001 TC2-B0FT-ORIG-S2-001 TC2-B0FT-ORIG-S3-003; do
        [ -f "$B/runs/$r/final-model.pt" ] || { echo "NOT SUBMITTED $NAME: missing $r/final-model.pt"; exit 1; }
        V="$V;$r|--checkpoint $B/runs/$r/final-model.pt"
      done
      submit "$JOB" hift_validate.sbatch --export=ALL,REPO="$R",VAL_JOBS="$V"; exit 0
    fi
    FT="--backbone alexnet --init official --epochs 10 --steps-per-epoch 1000"
    P1="--backbone alexnet --init official --layout corrected --epochs 10"   # stage 2 P1: forgetting / LR sweep
    VAL=""   # stage-2 runs hold out 10% of UAVDT videos (V3) and validate in-job
    S3="--init scratch-head --epochs 20 --steps-per-epoch 1000"
    case $NAME in  # job name | run id | training args (step 1 = B0 fine-tune; step 3 = scale augmentation test)
      hift_b0ft_orig) RUN=TC2-B0FT-ORIG-001; ARGS="$FT --layout original" ;;
      hift_b0ft_corr) RUN=TC2-B0FT-CORR-001; ARGS="$FT --layout corrected" ;;
      hift_s3_alex18) RUN=TC2-S3-ALEX-SC18-001; ARGS="$S3 --backbone alexnet --search-scale-aug 0.18" ;;
      hift_s3_alex40) RUN=TC2-S3-ALEX-SC40-001; ARGS="$S3 --backbone alexnet --search-scale-aug 0.40" ;;
      hift_s3_res18)  RUN=TC2-S3-RES18-SC18-001; ARGS="$S3 --backbone resnet18 --search-scale-aug 0.18 --resnet-weights $B/pretrained/resnet18-f37072fd.pth" ;;
      hift_s3_res40)  RUN=TC2-S3-RES18-SC40-001; ARGS="$S3 --backbone resnet18 --search-scale-aug 0.40 --resnet-weights $B/pretrained/resnet18-f37072fd.pth" ;;
      hift_p1_ref)    RUN=TC2-P1-REF-001;    ARGS="$P1 --steps-per-epoch 1000 --ft-lr 1e-3 1e-4"; VAL=1 ;;
      hift_p1_lr3e4)  RUN=TC2-P1-LR3E4-001;  ARGS="$P1 --steps-per-epoch 1000 --ft-lr 3e-4 3e-5"; VAL=1 ;;
      hift_p1_frozen) RUN=TC2-P1-FROZEN-001; ARGS="$P1 --steps-per-epoch 1000 --ft-lr 1e-3 1e-4 --ft-backbone frozen"; VAL=1 ;;
      hift_p1_long)   RUN=TC2-P1-LONG-001;   ARGS="$P1 --steps-per-epoch 3000 --ft-lr 1e-3 1e-5"; VAL=1 ;;
      *) echo "NOT SUBMITTED $NAME: unknown job"; exit 1 ;;
    esac
    if [ -n "$SEEDK" ]; then RUN=${RUN%-001}-S$SEEDK-001; ARGS="$ARGS --seed $((20261100+SEEDK)) --data-seed $((20261027+SEEDK))"; fi
    ATTEMPT=${3:-1}; case $ATTEMPT in [1-9]) RUN=${RUN%-001}-00$ATTEMPT ;; *) echo "NOT SUBMITTED $NAME: bad attempt $ATTEMPT"; exit 1 ;; esac
    [ -e "$B/runs/$RUN" ] && { echo "NOT SUBMITTED $JOB: run dir $RUN exists (never reused)"; exit 1; }
    case $NAME in hift_s3_res*) (cd "$B/pretrained" 2>/dev/null && sha256sum -c resnet18.sha256 >/dev/null) || { echo "NOT SUBMITTED $NAME: resnet weights missing/unverified"; exit 1; } ;; esac
    EXTRA=""; [ -n "$VAL" ] && EXTRA=",HOLDOUT=0.1,VALIDATE=1"
    submit "$JOB" hift_train.sbatch --export=ALL,REPO="$R",RUN_ID=$RUN,TRAIN_ARGS="$ARGS"$EXTRA ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
