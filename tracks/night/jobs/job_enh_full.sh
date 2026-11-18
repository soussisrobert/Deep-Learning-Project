#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=HiFT_enh_full
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E2: gamma_both, clahe_both, gamma+clahe both on ALL 180 NAT2021 sequences (see experiments_log.md).
# New "_180" tags so these never mix with the 30 sequence screening folders from E1.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml"

python test.py $COMMON --enh_template gamma       --enh_search gamma       --tag _gamma_both_180
python test.py $COMMON --enh_template clahe       --enh_search clahe       --tag _clahe_both_180
python test.py $COMMON --enh_template gamma_clahe --enh_search gamma_clahe --tag _gc_both_180

mkdir -p illumination
python eval_nat.py --dataset NAT2021 --compare --csv illumination/E2_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma_both_180 first1.2_clahe_both_180 first1.2_gc_both_180
