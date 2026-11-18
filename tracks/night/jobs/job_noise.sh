#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_noise
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Measure noise vs brightness on NAT2021 (6 frames per sequence) for calibrating the synthetic darkening. CPU only.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/measure_nat_noise.py 6 0.3
python -I tools/measure_nat_noise.py 6 0.5
python -I tools/measure_nat_noise.py 6 0.9
