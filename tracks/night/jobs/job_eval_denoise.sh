#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --job-name=HiFT_eval_dn
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E5 official evaluation of the three denoisers that finished (non-local means was dropped, see experiments_log.md).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
cd tools

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --compare --csv illumination/E5_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_gamma0.6 first1.2_dn_gauss3_g0.6 first1.2_dn_gauss5_g0.6 first1.2_dn_bilateral_g0.6
