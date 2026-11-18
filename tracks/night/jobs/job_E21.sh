#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:30:00
#SBATCH --job-name=HiFT_E21
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E21: test-time sweep of LR / PENALTY_K / WINDOW_INFLUENCE with first.pth + anchor 0.2 + gamma 0.6.
# A: 27 points on NATUD135VAL; the default point (0.30 / 0.08 / 0.42) must reproduce 0.4934 (stop otherwise); selection by the pre-fixed rule (E21_select.py).
# B: the selected point once on NAT2021, NAT2021L, NATUD135TEST; official tables, paired bootstrap, size diagnostics (E21_report.py). Finished runs (.done) are skipped.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools
C="--config ../experiments/config.yaml"
AG="--enh_template gamma --enh_search gamma --gamma 0.6 --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2"
run() {  # dataset lr pk wi
  F=first1.2_E21_lr$2_pk$3_wi$4
  if [ ! -f results/$1/$F/.done ]; then
    python test_tmpl_sweep.py --dataset $1 $C $AG --lr $2 --penalty_k $3 --window_influence $4 --tag _E21_lr$2_pk$3_wi$4
    touch results/$1/$F/.done
  fi
}
run NATUD135VAL 0.30 0.08 0.42
python -I illumination/E21_select.py check
for LR in 0.15 0.30 0.45; do for PK in 0.04 0.08 0.16; do for WI in 0.32 0.42 0.52; do
  run NATUD135VAL $LR $PK $WI
done; done; done
python -I illumination/E21_select.py select
read LR PK WI < illumination/E21_selected.txt
for DS in NAT2021 NAT2021L NATUD135TEST; do run $DS $LR $PK $WI; done
SEL=first1.2_E21_lr${LR}_pk${PK}_wi${WI}
python eval_nat.py --dataset NAT2021 --trackers first1.2_m_anchor $SEL --compare --csv illumination/E21_eval_NAT2021.csv
python eval_nat.py --dataset NAT2021L --trackers first1.2_m_anchor $SEL --compare --csv illumination/E21_eval_NAT2021L.csv
python eval_nat.py --dataset NATUD135TEST --trackers first1.2_ag $SEL --compare --csv illumination/E21_eval_NATUD135TEST.csv
python -I illumination/E21_report.py
