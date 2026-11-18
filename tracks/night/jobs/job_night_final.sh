#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --job-name=HiFT_night_final
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# usage: sbatch job_night_final.sh <A|B|C|D>
# Final evaluation of the checkpoint selected on the UAVDark135 val split (experiments/night_ft/runs/<run>/selected.txt):
# NAT2021, NAT2021L, UAVDark135 test, held-out UAV123 daytime set, each with no enhancement, gamma 0.6 on both crops, and anchor 0.2 + gamma 0.6;
# paired bootstrap against first.pth + anchor + gamma 0.6 (eval_nat.py --compare); then the E13 margin logging on NAT2021 (anchor + gamma 0.6).
set -e
RUN=$1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
# chain, so that the GPU does not idle when nobody is watching: queue the next job behind this one (needs sbatch to work on the compute node;
# if it does not, the chain stops here and is continued by hand). B -> C training, C -> D final evaluation.
case $RUN in
  B) sbatch --dependency=afterok:${SLURM_JOB_ID} job_night_train.sh C || echo "CHAIN: could not queue C training" ;;
  C) sbatch --dependency=afterok:${SLURM_JOB_ID} job_night_final.sh D || echo "CHAIN: could not queue D final evaluation" ;;
esac
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools
K=$(cat ../experiments/night_ft/runs/$RUN/selected.txt)
CK=../experiments/night_ft/runs/$RUN/snapshot/checkpoint_e$K.pth
echo "run $RUN selected epoch $K ($CK)"
C="--config ../experiments/config.yaml"
G="--enh_template gamma --enh_search gamma --gamma 0.6"
AG="--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 $G"
mkdir -p illumination
for DS in NATUD135TEST NATUAV123HELD NAT2021L NAT2021; do
  python test.py --dataset $DS --snapshot $CK $C --tag _${RUN}_none
  python test.py --dataset $DS --snapshot $CK $C $G --tag _${RUN}_g06
  python test_tmpl.py --dataset $DS --snapshot $CK $C $AG --tag _${RUN}_ag
  case $DS in
    NAT2021)  B="first1.2 first1.2_gamma0.6_180 first1.2_p15_a0.2_180" ;;
    NAT2021L) B="first1.2 first1.2_gamma0.6_L first1.2_p15_a0.2_L" ;;
    *)        B="first1.2_none first1.2_g06 first1.2_ag" ;;
  esac
  N="checkpoint_e${K}1.2_${RUN}_none checkpoint_e${K}1.2_${RUN}_g06 checkpoint_e${K}1.2_${RUN}_ag"
  echo "################ $DS: first.pth (none, gamma 0.6, anchor+gamma 0.6) then run $RUN epoch $K (same order)"
  python eval_nat.py --dataset $DS --trackers $B $N --compare --csv illumination/final_${RUN}_$DS.csv
done
python test_margin.py --dataset NAT2021 --snapshot $CK $C $AG --tag _m_${RUN}_ag
python -I margin_summary.py margin/NAT2021 first1.2_m_anchor checkpoint_e${K}1.2_m_${RUN}_ag
python -I margin_analysis.py --runs first1.2_m_anchor checkpoint_e${K}1.2_m_${RUN}_ag --out_csv illumination/E13_${RUN}.csv
