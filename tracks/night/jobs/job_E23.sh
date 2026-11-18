#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --job-name=HiFT_E23
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E23: per-video CSVs (official OPEBenchmark through the NATDataset adapter), cross-checks against eval_nat.py and the own formula, figures. CPU only, existing results.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
echo "== node"; hostname; sinfo -N -p <your-partition> -o "%N %G %c %m" | head -8; scontrol show node $(hostname -s) 2>/dev/null | grep -i -E "Gres|CPUTot|RealMemory" | head -4
echo "== du before"; du -sm night_evidence 2>/dev/null | tail -1
cd tools
python -I illumination/E23_package.py
echo "== du after"; du -sm ../night_evidence | tail -1; du -sm ../night_evidence/figures | tail -1
