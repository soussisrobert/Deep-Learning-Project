#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_ext_uavdt
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Phase 0: extract the three UAVDT zips (frames 6.8 GB, attributes, toolkit 0.8 GB uncompressed) into
# training_dataset/UAVDT/extracted/<part>/. The zips are kept. Estimated 8 GB written; home was 77.3 GB.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
echo "home before: $(du -sm ${HOME} | cut -f1) MB"
python -I tools/extract_zip_checked.py training_dataset/UAVDT/attributes/M_attr.zip training_dataset/UAVDT/extracted/attributes
python -I tools/extract_zip_checked.py training_dataset/UAVDT/toolkit/UAV-benchmark-MOTD_v1.0.zip training_dataset/UAVDT/extracted/toolkit
python -I tools/extract_zip_checked.py training_dataset/UAVDT/benchmark_M/UAV-benchmark-M.zip training_dataset/UAVDT/extracted/benchmark_M
echo "home after: $(du -sm ${HOME} | cut -f1) MB"
