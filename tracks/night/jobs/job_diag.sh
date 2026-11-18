#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=HiFT_diag
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/dhash_diag.py
