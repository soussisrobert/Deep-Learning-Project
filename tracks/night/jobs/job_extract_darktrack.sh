#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=HiFT_ext_dark
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Extract DarkTrack2021 (3-part split zip, 21.4 GB) into training_dataset/DarkTrack2021/extracted. Zips are kept.
# Refuses to start if the extracted size would push home above 98 GB.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
E=training_dataset/extra_drive
python -I tools/extract_split_zip.py --parts $E/DarkTrack2021.zip.001 $E/DarkTrack2021.zip.002 $E/DarkTrack2021.zip.003 \
    --out training_dataset/DarkTrack2021/extracted --max_home_mb 98000
du -sm ${HOME} | cut -f1
