#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_ext_uav123
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Extract Dataset_UAV123.zip (14.1 GB uncompressed, 113,766 files) into training_dataset/UAV123/extracted/.
# The zip is kept. Disk guard: stop if the home directory is above 70 GB.
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
used=$(du -sm ${HOME} 2>/dev/null | cut -f1)
echo "home usage before: ${used} MB"
if [ "$used" -gt 70000 ]; then echo "home above 70 GB, stopping"; exit 2; fi
python -I tools/extract_zip_checked.py training_dataset/UAV123/Dataset_UAV123.zip training_dataset/UAV123/extracted
du -sm ${HOME} | cut -f1
