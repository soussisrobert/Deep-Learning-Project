#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=HiFT_marginan
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E13 analysis (see experiments_log.md): needs the margin logs and boxes written by job_margin.sh. CPU only.
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}/tools
python -I margin_analysis.py --key mx --out_csv illumination/E13_split_events.csv
echo
echo "################ same analysis with the grid-distance second peak (key m) for comparison"
python -I margin_analysis.py --key m --out_csv illumination/E13_split_events_gridpeak.csv
