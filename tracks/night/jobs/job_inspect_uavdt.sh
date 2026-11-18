#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=HiFT_insp_uavdt
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# List the contents of the three UAVDT zips without extracting anything (CPU only).
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
python -I tools/inspect_uavdt.py
