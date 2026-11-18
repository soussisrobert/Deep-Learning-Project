#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:30:00
#SBATCH --job-name=HiFT_bank
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E18: oracle template bank (upper bound). first.pth, gamma 0.6 on both crops, NO anchored update.
# R0 = bank of one (K = 1) on NAT2021L then NAT2021, checked against the stored gamma-only results (stop if the mean success AUC differs by > 0.002);
# R1 = oracle bank, thr 0.52, K = 5, on NAT2021L then NAT2021; then the official tables / paired bootstraps (eval_nat.py) and tools/illumination/E18_analysis.py.
# Finished folders (those with .done) are skipped, so the job can be resubmitted.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools
C="--config ../experiments/config.yaml"
G="--enh_template gamma --enh_search gamma --gamma 0.6"
run() {  # dataset K tag
  if [ ! -f results/$1/first1.2$3/.done ]; then
    python test_bank.py --dataset $1 $C $G --bank_k $2 --bank_thr 0.52 --tag $3
    touch results/$1/first1.2$3/.done
  fi
}
run NAT2021L 1 _bank_K1
run NAT2021 1 _bank_K1
python -I illumination/E18_analysis.py check NAT2021L NAT2021
run NAT2021L 5 _bank_K5
run NAT2021 5 _bank_K5
for DS in NAT2021 NAT2021L; do
  if [ $DS = NAT2021 ]; then S=first1.2_gamma0.6_180; else S=first1.2_gamma0.6_L; fi
  python eval_nat.py --dataset $DS --trackers $S first1.2_bank_K1 first1.2_bank_K5 first1.2_m_anchor --compare --csv illumination/E18_eval_$DS.csv
done
python -I illumination/E18_analysis.py report
