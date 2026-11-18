#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_anchor180
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E10 (see experiments_log.md): the anchored template update from E8 (a = 0.2, feature average of the last 3 frames,
# no gate) with gamma 0.6 on both crops, on ALL 180 NAT2021 sequences, compared with the baseline and with
# gamma 0.6 alone (first1.2_gamma0.6_180 is the E4 run, same settings, frame 1 template only).
# Template update: template = 0.8 * frame 1 features + 0.2 * mean of the last 3 features (frame 1 included at the start).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

python test_tmpl.py --dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml \
    --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _p15_a0.2_180

python eval_nat.py --dataset NAT2021 --compare --csv illumination/E10_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma0.6_180 first1.2_p15_a0.2_180
