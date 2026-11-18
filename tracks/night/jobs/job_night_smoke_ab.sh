#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=00:40:00
#SBATCH --job-name=HiFT_night_smoke_ab
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Short smoke checks of runs A, B and D (100 iterations each, no checkpoint written, 3 validation batches). C was checked in job 49143.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
for R in A B D; do
  echo "######## run $R"
  python -I tools/train_night.py --cfg experiments/night_ft/config.yaml experiments/night_ft/run_$R.yaml --run smoke_$R --max_iters 100 --val_batches 3
done
