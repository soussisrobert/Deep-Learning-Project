#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --job-name=HiFT_worse
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E3a (see experiments_log.md): re-run only the sequences where gamma_both (gamma 0.4) was clearly worse
# than the baseline (delta success AUC < -0.01, 60 sequences) with several gamma / CLAHE-clip pairs.
# Every run uses gamma+CLAHE on both template and search crop.
# The last run is a near-identity control (gamma 1.0, clip 1.0): it shows how much results move
# on these sequences from a tiny input change alone.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

LIST=illumination/worse_list_gamma_both.txt
tr -d '\r' < illumination/improvement_first1.2_gamma_both_180.csv | awk -F, 'NR>1 && $NF=="worse"{print $1}' > $LIST
echo "worse sequences: $(wc -l < $LIST)"

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list $LIST --enh_template gamma_clahe --enh_search gamma_clahe"

python test.py $COMMON --gamma 0.6  --clip 2.0 --tag _w_g0.6_c2.0
python test.py $COMMON --gamma 0.5  --clip 2.0 --tag _w_g0.5_c2.0
python test.py $COMMON --gamma 0.45 --clip 3.0 --tag _w_g0.45_c3.0
python test.py $COMMON --gamma 0.5  --clip 4.0 --tag _w_g0.5_c4.0
python test.py $COMMON --gamma 1.0  --clip 1.0 --tag _w_control_g1.0_c1.0

python eval_nat.py --dataset NAT2021 --video_list $LIST --compare --csv illumination/E3a_worse_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma_both_180 first1.2_clahe_both_180 first1.2_gc_both_180 \
    first1.2_w_control_g1.0_c1.0 first1.2_w_g0.6_c2.0 first1.2_w_g0.5_c2.0 first1.2_w_g0.45_c3.0 first1.2_w_g0.5_c4.0
