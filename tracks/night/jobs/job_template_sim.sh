#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_tmpl_sim
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E17: template-feature similarity to frame 1 at every 10th NAT2021 frame (inference only, first.pth, gamma 0.6), then the CPU analysis against first_ag results.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd tools
[ -f illumination/E17_template_sim.csv ] || python template_sim.py
python -I template_sim_analysis.py
