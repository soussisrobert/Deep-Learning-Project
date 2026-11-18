#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --job-name=HiFT_night_train
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# usage: sbatch job_night_train.sh <A|B|C|D>
# 1) fine-tune 10 epochs (330k pairs each, checkpoint + val loss per epoch; stops after an epoch if the 4 h budget would be exceeded)
# 2) tracking success AUC on the UAVDark135 val split for EVERY epoch checkpoint, with no enhancement and with anchor 0.2 + gamma 0.6;
#    first.pth is evaluated once for reference; the epoch with the highest mean success AUC of the anchor + gamma 0.6 curve is selected.
set -e
RUN=$1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
# chain (see job_night_final.sh): C training -> D training, D training -> C final evaluation
case $RUN in
  C) sbatch --dependency=afterok:${SLURM_JOB_ID} job_night_train.sh D || echo "CHAIN: could not queue D training" ;;
  D) sbatch --dependency=afterok:${SLURM_JOB_ID} job_night_final.sh C || echo "CHAIN: could not queue C final evaluation" ;;
esac
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
python -I tools/train_night.py --cfg experiments/night_ft/config.yaml experiments/night_ft/run_$RUN.yaml --run $RUN --time_budget_s 14400

cd tools
C="--config ../experiments/config.yaml"
AG="--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 --enh_template gamma --enh_search gamma --gamma 0.6"
if [ ! -f results/NATUD135VAL/first1.2_ag/.done ]; then
  python test.py --dataset NATUD135VAL --snapshot snapshot/first.pth $C --tag _none
  python test_tmpl.py --dataset NATUD135VAL --snapshot snapshot/first.pth $C $AG --tag _ag
  touch results/NATUD135VAL/first1.2_ag/.done
fi
TR="first1.2_none first1.2_ag"
for k in 1 2 3 4 5 6 7 8 9 10; do
  CK=../experiments/night_ft/runs/$RUN/snapshot/checkpoint_e$k.pth
  [ -f $CK ] || continue
  python test.py --dataset NATUD135VAL --snapshot $CK $C --tag _${RUN}_none_e$k
  python test_tmpl.py --dataset NATUD135VAL --snapshot $CK $C $AG --tag _${RUN}_ag_e$k
  TR="$TR checkpoint_e${k}1.2_${RUN}_none_e$k checkpoint_e${k}1.2_${RUN}_ag_e$k"
done
mkdir -p illumination
python eval_nat.py --dataset NATUD135VAL --trackers $TR --csv illumination/val_curve_$RUN.csv
python -I select_epoch.py illumination/val_curve_$RUN.csv $RUN _${RUN}_none,_${RUN}_ag _${RUN}_ag
