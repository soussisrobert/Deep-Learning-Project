#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_extract_train
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Extract the NAT2021 train set (4 raw split parts of one zip64 archive) into training_dataset/NAT2021_train/extracted/.
# The zips are kept. The user asked to ignore the 100 GB quota for this (home is about 92 GB before it starts,
# the extraction adds roughly the size of the zips). No GPU needed.

set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
T=training_dataset/NAT2021_train
echo "home usage before: $(du -sm ${HOME} | cut -f1) MB"

python -I tools/extract_split_zip.py \
    --parts $T/NAT2021_train.zip.001 $T/NAT2021_train.zip.002 $T/NAT2021_train.zip.003 $T/NAT2021_train.zip.004 \
    --out $T/extracted

echo "home usage after: $(du -sm ${HOME} | cut -f1) MB"
ls $T/extracted | head
