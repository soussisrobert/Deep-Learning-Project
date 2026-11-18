#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=HiFT_overnight
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Unattended chain (see experiments_log.md), replaces the separate jobs 48617 (E9), job_anchor180.sh and
# job_download_train.sh because the account is limited to 2 submitted jobs:
#   0. install gdown into the HiFT env if missing (before any tracking starts, so no environment changes during runs)
#   1. start the NAT2021 train set download in the background (zips only, not extracted, 85 GB home guard)
#   2. E9: gamma 0.6 (both crops, frame 1 template) on all 23 NAT2021L sequences + evaluation
#   3. E10: anchored template update (a = 0.2, feature average of 3, gamma 0.6) on all 180 NAT2021 sequences + evaluation
#   4. wait for the download to finish

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

# 0 + 1: download in the background
python -c "import gdown" 2>/dev/null || conda install -y -n HiFT -c conda-forge gdown || echo "gdown install failed, the download will be skipped"

DEST=${HIFT_ROOT}/training_dataset/NAT2021_train
mkdir -p $DEST

get() {
    local id=$1 name=$2
    local used=$(du -sm ${HOME} 2>/dev/null | cut -f1)
    echo "$(date) home usage before $name: ${used} MB"
    if [ "$used" -gt 85000 ]; then echo "home above 85 GB, stopping before $name"; exit 2; fi
    if [ -s "$DEST/$name" ]; then echo "$name already exists, skipping"; return; fi
    gdown "https://drive.google.com/uc?id=$id" -O "$DEST/$name.part" && mv "$DEST/$name.part" "$DEST/$name" || { echo "download of $name failed"; exit 1; }
    ls -l "$DEST/$name"
}

if python -c "import gdown" 2>/dev/null; then
    (
        get 1SR5PH6frXBpJwgvyDA_KLVDigEjaOpBj NAT2021_train.zip.001
        get 1GYmCsdIizGVEpUe75x9IWRLCATrpczkf NAT2021_train.zip.002
        get 1fHO1-yBi_4ZZfnALrZf8xgKCOdCVV9mY NAT2021_train.zip.003
        get 1VfwUj9OuN6ePTowVmLJNiOo4YHGtmx-M NAT2021_train.zip.004
        echo "$(date) download finished"
        ls -l $DEST
    ) > output/download_train_${SLURM_JOB_ID}.log 2>&1 &
    DL_PID=$!
else
    DL_PID=""
fi

cd tools

# 2: E9
echo "=== E9: gamma 0.6 on NAT2021L ==="
python test.py --dataset NAT2021L --snapshot snapshot/first.pth --config ../experiments/config.yaml \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _gamma0.6_L
python eval_nat.py --dataset NAT2021L --compare --csv illumination/E9_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma0.6_L

# 3: E10
echo "=== E10: anchored update a=0.2 on NAT2021 (180) ==="
python test_tmpl.py --dataset NAT2021 --snapshot snapshot/first.pth --config ../experiments/config.yaml \
    --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _p15_a0.2_180
python eval_nat.py --dataset NAT2021 --compare --csv illumination/E10_per_sequence_auc.csv \
    --trackers first1.2 first1.2_gamma0.6_180 first1.2_p15_a0.2_180

# 4
if [ -n "$DL_PID" ]; then
    echo "=== waiting for the download ==="
    wait $DL_PID
    cat output/download_train_${SLURM_JOB_ID}.log
fi
echo "=== all done ==="
