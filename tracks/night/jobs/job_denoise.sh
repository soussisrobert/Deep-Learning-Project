#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_denoise
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E5 (see experiments_log.md): denoise BEFORE gamma 0.6, on both template and search crop.
# Screening on the 40 sequence balanced sample from E3c (20 from the half that improved at gamma 0.4,
# 20 from the half that got worse). Reference runs already on disk for the same 40 sequences:
# first1.2 (baseline, 0.380) and first1.2_f_gamma0.6 (gamma 0.6 alone, 0.452).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list illumination/E3c_all_list.txt --enh_template gamma --enh_search gamma --gamma 0.6"

for D in gauss3 gauss5 bilateral nlm; do
    python test.py $COMMON --denoise $D --tag _dn_${D}_g0.6
done

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --compare --csv illumination/E5_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_gamma0.6 first1.2_dn_gauss3_g0.6 first1.2_dn_gauss5_g0.6 \
    first1.2_dn_bilateral_g0.6 first1.2_dn_nlm_g0.6
