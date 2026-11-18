#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_tmpl
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E7 (see experiments_log.md): rebuild the template after every frame from the last 3 frames.
#   pixel   = average the last 3 template crops, then run the backbone (the user's idea)
#   feature = run the backbone on each crop and average the last 3 feature sets
# Frame 1 is included until it drops out of the window. No confidence gate, no frame 1 anchor.
# Balanced 40 sequence sample from E3c. For each variant: no enhancement, gamma 0.95 (almost no change), gamma 0.6.
# References with the frame 1 template already exist: first1.2, first1.2_f_gamma0.95, first1.2_f_gamma0.6.
# Uses tools/test_tmpl.py (a copy of test.py with the template tracker) so the running E6 job was not touched.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list illumination/E3c_all_list.txt --tmpl_k 3"

for M in pixel feature; do
    S=${M:0:3}; [ $M = feature ] && S=feat
    python test_tmpl.py $COMMON --tmpl_update $M --tag _t_${S}_base
    python test_tmpl.py $COMMON --tmpl_update $M --enh_template gamma --enh_search gamma --gamma 0.95 --tag _t_${S}_g0.95
    python test_tmpl.py $COMMON --tmpl_update $M --enh_template gamma --enh_search gamma --gamma 0.6  --tag _t_${S}_g0.6
done

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --csv illumination/E7_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_gamma0.95 first1.2_f_gamma0.6 \
    first1.2_t_pix_base first1.2_t_pix_g0.95 first1.2_t_pix_g0.6 \
    first1.2_t_feat_base first1.2_t_feat_g0.95 first1.2_t_feat_g0.6

python tmpl_analysis.py
