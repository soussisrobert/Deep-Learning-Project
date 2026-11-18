#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=HiFT_selattr
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E12 (see experiments_log.md): selection check for gamma (choose on even-indexed sequences, report on odd, and the
# reverse) and per-attribute / per-darkness breakdown of the gains. Uses only the existing per-sequence AUC files, no
# tracking. CPU only.
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}/tools
python -I selection_attr_analysis.py
