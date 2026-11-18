#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=HiFT_margin
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E13 (see experiments_log.md): log the top-1 and top-2 peak of the final score map for every frame, on all 180
# NAT2021 sequences, for three settings whose boxes already exist: baseline (E0), gamma 0.6 (E4), anchor 0.2 +
# gamma 0.6 (E10). The tracker is deterministic, so the boxes of these runs must be identical to the existing result
# folders (checked afterwards with cmp); that is the proof that the logging copy of track() changes nothing.
# A smoke test on 3 sequences runs first (set -e stops the job there if it fails).
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

C="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml"

python test_margin.py $C --every 60 --tag _m_smoke
head -3 margin/NAT2021/first1.2_m_smoke/N01001.csv
cmp results/NAT2021/first1.2_m_smoke/N01001.txt results/NAT2021/first1.2/N01001.txt
echo "smoke: boxes identical to E0 for N01001"

python test_margin.py $C --tag _m_base
python test_margin.py $C --enh_template gamma --enh_search gamma --gamma 0.6 --tag _m_g0.6
python test_margin.py $C --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _m_anchor

echo "box identity check against the earlier runs (number of sequence files that differ):"
for pair in "_m_base first1.2" "_m_g0.6 first1.2_gamma0.6_180" "_m_anchor first1.2_p15_a0.2_180"; do
    set -- $pair
    n=0
    for f in results/NAT2021/first1.2$1/*.txt; do
        cmp -s "$f" "results/NAT2021/$2/$(basename $f)" || n=$((n+1))
    done
    echo "$1 vs $2: $n differ of $(ls results/NAT2021/first1.2$1/*.txt | wc -l)"
done
