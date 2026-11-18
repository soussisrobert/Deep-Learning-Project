#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_nat2021L
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E9 (see experiments_log.md): the best setting from NAT2021 (gamma 0.6 on both template and search crop, frame 1
# template) on all 23 NAT2021L sequences. Baseline on NAT2021L already exists: tools/results/NAT2021L/first1.2.
# Note: 13 of the 23 NAT2021L sequences start from the same first box as a NAT2021 sequence, so NAT2021L is not
# fully independent of NAT2021.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

python test.py --dataset NAT2021L --snapshot snapshot/first.pth --config ../experiments/config.yaml \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _gamma0.6_L

python eval_nat.py --dataset NAT2021L --compare --csv illumination/E9_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma0.6_L
