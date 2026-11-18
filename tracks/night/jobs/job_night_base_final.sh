#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --job-name=HiFT_night_bf
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# usage: sbatch job_night_base_final.sh <run>: first.pth baselines on the new test sets (skipped when already done), then the final evaluation of <run>.
set -e
cd ${HIFT_ROOT}
bash job_night_base.sh
bash job_night_final.sh $1
