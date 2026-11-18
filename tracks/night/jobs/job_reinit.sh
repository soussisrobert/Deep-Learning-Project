#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_reinit
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E6 (see experiments_log.md): break the feedback loop. Every N frames the tracker's box and scale are reset
# to the ground truth (original template kept). Diagnostic only, it uses the ground truth.
# Balanced 40 sequence sample from E3c. 'no reset' runs already exist: first1.2, first1.2_f_gamma0.95, first1.2_f_gamma0.6.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list illumination/E3c_all_list.txt"

for N in 200 50; do
    python test.py $COMMON --reinit_every $N --tag _r${N}_base
    python test.py $COMMON --reinit_every $N --enh_template gamma --enh_search gamma --gamma 0.95 --tag _r${N}_g0.95
    python test.py $COMMON --reinit_every $N --enh_template gamma --enh_search gamma --gamma 0.6  --tag _r${N}_g0.6
done

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --csv illumination/E6_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_gamma0.95 first1.2_f_gamma0.6 \
    first1.2_r200_base first1.2_r200_g0.95 first1.2_r200_g0.6 \
    first1.2_r50_base first1.2_r50_g0.95 first1.2_r50_g0.6

python reinit_analysis.py
