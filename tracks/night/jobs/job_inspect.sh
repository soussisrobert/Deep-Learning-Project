#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=HiFT_inspect
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}/tools
export PYTHONUNBUFFERED=1

python pixel_inspect.py --seq N01003 --frame 1      # very dark, HiFT AUC 0.114
python pixel_inspect.py --seq N01003 --frame 600    # same sequence, mid-way
python pixel_inspect.py --seq N02002 --frame 1      # brighter (mean gray 44), AUC 0.779
