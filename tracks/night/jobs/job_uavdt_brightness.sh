#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --job-name=HiFT_uavdt_br
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Brightness of the UAVDT-M sequences (every 20th frame, read from the zip, nothing extracted) next to the
# daylight / night / fog label and the NAT2021 test brightness. CPU only.
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
python -I tools/uavdt_brightness.py
