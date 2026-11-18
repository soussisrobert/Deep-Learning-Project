#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH --job-name=HiFT_E22
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E22: loss taxonomy on NAT2021 (A-C) and contact sheet (D). CPU only, existing results, no tracking. Home usage is printed before and after.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}/tools
echo "du -sm ~ before:"; du -sm ~ 2>/dev/null | tail -1
python -I illumination/E22_loss_taxonomy.py
python -I illumination/E22_contact_sheet.py
echo "du -sm ~ after:"; du -sm ~ 2>/dev/null | tail -1
du -sm illumination/E22_contact | tail -1
