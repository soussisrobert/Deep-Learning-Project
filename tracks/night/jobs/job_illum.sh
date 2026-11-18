#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_illum
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}/tools
export PYTHONUNBUFFERED=1

for NAME in NAT2021 NAT2021L; do
    python illumination.py --dataset $NAME --stride 5 --procs 4
done
