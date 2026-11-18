#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_anchorL
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E11 (see experiments_log.md): the anchored template update from E8/E10 (a = 0.2, feature average of the last 3
# frames, no gate) with gamma 0.6 on both crops, on all 23 NAT2021L sequences. This is the third condition of the
# decision rule written in E10 (the anchored update must not lose on NAT2021L). Compared with the baseline and with
# gamma 0.6 alone (E9, first1.2_gamma0.6_L).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

python test_tmpl.py --dataset NAT2021L --snapshot snapshot/first.pth --config ../experiments/config.yaml \
    --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _p15_a0.2_L

python eval_nat.py --dataset NAT2021L --compare --csv illumination/E11_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma0.6_L first1.2_p15_a0.2_L
