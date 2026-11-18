#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH --job-name=HiFT_E22D
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E22 step D only (the A-C part of job 55277 finished; D crashed in the geometry check, fixed).
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}/tools


python -I illumination/E22_contact_sheet.py
echo "du -sm ~ after:"; du -sm ~ 2>/dev/null | tail -1
du -sm illumination/E22_contact | tail -1
