#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_p15
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E8 (see experiments_log.md): P15 follow-up to E7. Template update from the last 3 frames (feature average) with
#   an anchor on the frame 1 template: template = (1 - a) * frame 1 + a * mean(last 3), a = 0.5 and 0.2
#   a confidence gate: update only when the best score >= median of the earlier scores of the sequence
# All on both crops with gamma 0.6 (best setting), balanced 40 sequence sample from E3c.
# References on the same sequences: first1.2_f_gamma0.6 (frame 1 template) and first1.2_t_feat_g0.6 (E7, no anchor, no gate).

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list illumination/E3c_all_list.txt --tmpl_update feature --tmpl_k 3 --enh_template gamma --enh_search gamma --gamma 0.6"

python test_tmpl.py $COMMON --tmpl_alpha 0.5                      --tag _p15_a0.5
python test_tmpl.py $COMMON --tmpl_alpha 0.2                      --tag _p15_a0.2
python test_tmpl.py $COMMON --tmpl_alpha 0.5 --tmpl_gate median   --tag _p15_a0.5_gate
python test_tmpl.py $COMMON --tmpl_alpha 0.2 --tmpl_gate median   --tag _p15_a0.2_gate
python test_tmpl.py $COMMON --tmpl_gate median                    --tag _p15_gate_only

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --compare --csv illumination/E8_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_gamma0.6 first1.2_t_feat_g0.6 \
    first1.2_p15_a0.5 first1.2_p15_a0.2 first1.2_p15_a0.5_gate first1.2_p15_a0.2_gate first1.2_p15_gate_only
