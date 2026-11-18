#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_enh
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Enhancement experiments on a 30 sequence subset of NAT2021 (every 6th sequence, sorted by name).
# The baseline (tools/results/NAT2021/first1.2) already exists for all 180 sequences and is
# restricted to the same subset by eval_nat.py --every 6.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --every 6"

python test.py $COMMON --enh_template gamma       --enh_search gamma       --tag _gamma_both
python test.py $COMMON --enh_template clahe       --enh_search clahe       --tag _clahe_both
python test.py $COMMON --enh_template gamma_clahe --enh_search gamma_clahe --tag _gc_both
python test.py $COMMON --enh_template gamma_clahe --enh_search none        --tag _gc_template_only
python test.py $COMMON --enh_template none        --enh_search gamma_clahe --tag _gc_search_only

python eval_nat.py --dataset NAT2021 --every 6
