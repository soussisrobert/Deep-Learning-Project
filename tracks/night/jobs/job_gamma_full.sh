#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_gamma_full
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E4 (see experiments_log.md): gamma-only enhancement at 0.8 and 0.6 on both template and search crop,
# on all 180 NAT2021 sequences, compared with the baseline and the existing gamma 0.4 run.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --enh_template gamma --enh_search gamma"

python test.py $COMMON --gamma 0.8 --tag _gamma0.8_180
python test.py $COMMON --gamma 0.6 --tag _gamma0.6_180

python eval_nat.py --dataset NAT2021 --compare --csv illumination/E4_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma_both_180 first1.2_gamma0.8_180 first1.2_gamma0.6_180
