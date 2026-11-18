#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --job-name=HiFT_fragility
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E3c (see experiments_log.md): is HiFT's per-sequence result fragile?
# Random sample of 20 sequences from the half that IMPROVED with gamma 0.4 and 20 from the half that got WORSE
# (groups from improvement_first1.2_gamma_both_180.csv, tolerance +/-0.01), fixed random seed.
# Runs: a repeat of the baseline (nothing changed), gamma-only at 0.95 / 0.9 / 0.8 / 0.6 / 0.5,
# CLAHE-only at clip 1 / 2 / 4, each on both template and search crop.
# Existing full-set runs gamma 0.4 (first1.2_gamma_both_180) and CLAHE clip 3 (first1.2_clahe_both_180)
# are reused for the same sequences.

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools

CSV=illumination/improvement_first1.2_gamma_both_180.csv
tr -d '\r' < $CSV | awk -F, 'NR>1 && $NF=="improved"{print $1}' | shuf -n 20 --random-source=<(yes 42) | sort > illumination/E3c_improved_list.txt
tr -d '\r' < $CSV | awk -F, 'NR>1 && $NF=="worse"{print $1}'    | shuf -n 20 --random-source=<(yes 42) | sort > illumination/E3c_worse_list.txt
cat illumination/E3c_improved_list.txt illumination/E3c_worse_list.txt | sort > illumination/E3c_all_list.txt
echo "improved: $(wc -l < illumination/E3c_improved_list.txt)  worse: $(wc -l < illumination/E3c_worse_list.txt)  all: $(wc -l < illumination/E3c_all_list.txt)"

COMMON="--dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml --video_list illumination/E3c_all_list.txt"

python test.py $COMMON --tag _f_repeat_baseline

for G in 0.95 0.9 0.8 0.6 0.5; do
    python test.py $COMMON --enh_template gamma --enh_search gamma --gamma $G --tag _f_gamma$G
done

for C in 1.0 2.0 4.0; do
    python test.py $COMMON --enh_template clahe --enh_search clahe --clip $C --tag _f_clahe$C
done

python eval_nat.py --dataset NAT2021 --video_list illumination/E3c_all_list.txt --csv illumination/E3c_per_sequence_auc.csv \
    --trackers first1.2 first1.2_f_repeat_baseline first1.2_f_gamma0.95 first1.2_f_gamma0.9 first1.2_f_gamma0.8 \
    first1.2_f_gamma0.6 first1.2_f_gamma0.5 first1.2_gamma_both_180 \
    first1.2_f_clahe1.0 first1.2_f_clahe2.0 first1.2_clahe_both_180 first1.2_f_clahe4.0

python fragility_analysis.py
