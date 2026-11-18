#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_per_seq
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E14: per-sequence analysis of B_ag and A_ag against first_ag (CPU work only; the partition requires a GPU slot, which is not used).
# Reads existing results, margin logs and CSVs; writes tools/illumination/E14_*.csv and E14_fig*.png.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}/tools
python -I per_seq_analysis.py
