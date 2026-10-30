#!/bin/bash
# Run a queue file: one run per line, "<run name><TAB><command>".
# Skips runs whose results/<run name>/summary.json exists; keeps going if one run fails;
# merges finished split conditions and refreshes the tables at the end.
#   bash scripts/run_queue.sh plan/queue/01_gpu_P1.tsv
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE=${1:?queue file}
OUT=${OUT:-results}
failed=0
while IFS=$'\t' read -r name cmd; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "[skip] $name (done)"; continue; fi
  if [[ "$name" =~ ^(.+)__p[0-9]+$ && -f "$OUT/${BASH_REMATCH[1]}/summary.json" ]]; then
    echo "[skip] $name (merged ${BASH_REMATCH[1]} exists)"; continue; fi
  echo "[$(date +%H:%M:%S)] start $name"
  if eval "$cmd"; then echo "[$(date +%H:%M:%S)] done  $name"
  else echo "[$(date +%H:%M:%S)] FAILED $name"; failed=$((failed + 1)); rm -rf "$OUT/$name"; fi
done < "$QUEUE"
python scripts/merge_all.py "$OUT" || true
python scripts/summarize.py --results "$OUT" || true
echo "queue $QUEUE finished, $failed failed"
exit $failed
