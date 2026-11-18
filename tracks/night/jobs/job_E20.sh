#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=HiFT_E20
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E20: box-size oracle on saved first_ag boxes (CPU only, no tracking); first1.2_ag vs m_anchor folder check first.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}/tools
python -I illumination/E20_size_oracle.py
