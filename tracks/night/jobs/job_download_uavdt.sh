#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --job-name=HiFT_dl_uavdt
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Download UAVDT (links given by the user, 2026-10-29) into training_dataset/UAVDT/, one sub folder per file because the
# file names and archive types are not known before the download. Archives are NOT extracted here: look at them first.
#   benchmark_M  UAVDT-Benchmark-M dataset (6.32 GB)
#   attributes   attribute file (daylight / night / fog labels per sequence, to pick the night sequences)
#   toolkit      DET / MOT toolkit (train / test split lists)
# No GPU needed (gres=gpu:0). Disk guard: stop before a file if the home directory is above 85 GB (quota about 100 GB).
set -e

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

# gdown is installed into the HiFT env only
python -c "import gdown" 2>/dev/null || conda install -y -n HiFT -c conda-forge gdown

DEST=${HIFT_ROOT}/training_dataset/UAVDT
mkdir -p $DEST

get() {
    local id=$1 sub=$2
    local used=$(du -sm ${HOME} 2>/dev/null | cut -f1)
    echo "home usage before $sub: ${used} MB"
    if [ "$used" -gt 85000 ]; then echo "home above 85 GB, stopping before $sub"; exit 2; fi
    mkdir -p $DEST/$sub
    if [ -n "$(ls -A $DEST/$sub)" ]; then echo "$sub is not empty, skipping (empty it to download again)"; return; fi
    # a trailing slash makes gdown keep the file name from Drive
    gdown "https://drive.google.com/uc?id=$id" -O "$DEST/$sub/"
    ls -l $DEST/$sub
}

get 1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1 attributes
get 19498uJd7T9w4quwnQEy62nibt3uyT9pq toolkit
get 1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc benchmark_M

echo "done"
for f in $DEST/*/*; do echo "$f: $(file -b "$f" | cut -c1-80)"; done
du -sm $DEST | cut -f1
du -sm ${HOME} | cut -f1
