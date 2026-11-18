#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=HiFT_E23a
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E23 step 1: official-setup check (checkpoint hashes, config history). Read-only, no python.
cd ${HIFT_ROOT}
echo "== sha256"; sha256sum tools/snapshot/first.pth; ls -l tools/snapshot; ls tools/snapshot/general_model.pth 2>&1 | head -1; find . -name "general_model.pth" 2>/dev/null | head
echo "== git"; git remote -v; git branch --show-current; git rev-parse HEAD
echo "== config.yaml status vs HEAD"; git status --short experiments/config.yaml; git diff --stat HEAD -- experiments/config.yaml
echo "== config.yaml commits"; git log --format='%h %ad %an %s' --date=short -- experiments/config.yaml
echo "== config.yaml at each commit (TRACK block)"
for c in $(git log --format=%h -- experiments/config.yaml); do echo "-- $c"; git show $c:experiments/config.yaml | sed -n '/^TRACK:/,/^TRAIN:/p' | grep -v "^$"; done
echo "== full diff of config.yaml between its first and last commit"; first=$(git log --format=%h -- experiments/config.yaml | tail -1); git diff $first HEAD -- experiments/config.yaml
echo "== config.py TRACK block at last commit (b66d3eb) and its diff to the working tree"; git show HEAD:pysot/core/config.py | sed -n '/__C.TRACK = CN/,/^__C.TRACK.LARGER/p'; git diff HEAD -- pysot/core/config.py
echo "== other copies of config.yaml on disk (HiFT tree)"; find ${HIFT_ROOT}/.. -name "config.yaml" -not -path "*/results/*" 2>/dev/null | head
echo "== ope_benchmark / statistics history"; git log --format='%h %ad %s' --date=short -- toolkit/evaluation/ope_benchmark.py toolkit/utils/statistics.py; git status --short toolkit/evaluation toolkit/utils
