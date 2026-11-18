#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=HiFT_lost_diag
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E15: re-detection diagnostic. The tracker is not changed. The E13 score logs of first.pth + anchor 0.2 + gamma 0.6 already exist for NAT2021
# (margin/NAT2021/first1.2_m_anchor); they are created here, with the same flags as job_night_final.sh uses, for NATUD135VAL and NAT2021L
# (GPU, a few minutes each), then tools/lost_diag.py (CPU) does the analysis.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools
C="--config ../experiments/config.yaml"
AG="--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 --enh_template gamma --enh_search gamma --gamma 0.6"
for DS in NATUD135VAL NAT2021L; do
  if [ ! -f margin/$DS/first1.2_m_anchor/.done ]; then
    python test_margin.py --dataset $DS --snapshot snapshot/first.pth $C $AG --tag _m_anchor
    touch margin/$DS/first1.2_m_anchor/.done
  fi
done
python -I lost_diag.py
