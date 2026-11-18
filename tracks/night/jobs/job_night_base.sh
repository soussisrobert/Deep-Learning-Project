#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --job-name=HiFT_night_base
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# first.pth on the three new evaluation sets with the three test-time settings (the NAT2021 / NAT2021L results already exist).
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
AG="--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 $G"
for DS in NATUD135VAL NATUD135TEST NATUAV123HELD; do
  [ -f results/$DS/first1.2_none/.done ] || { python test.py --dataset $DS --snapshot snapshot/first.pth $C --tag _none; touch results/$DS/first1.2_none/.done; }
  [ -f results/$DS/first1.2_g06/.done ] || { python test.py --dataset $DS --snapshot snapshot/first.pth $C $G --tag _g06; touch results/$DS/first1.2_g06/.done; }
  [ -f results/$DS/first1.2_ag/.done ] || { python test_tmpl.py --dataset $DS --snapshot snapshot/first.pth $C $AG --tag _ag; touch results/$DS/first1.2_ag/.done; }
  python eval_nat.py --dataset $DS --trackers first1.2_none first1.2_g06 first1.2_ag
done
