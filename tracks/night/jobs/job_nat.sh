#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=HiFT_NAT
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

cd ${HIFT_ROOT}
export PYTHONPATH=$PWD:$PYTHONPATH
DATA=test_dataset

test -f tools/snapshot/first.pth || { echo "missing tools/snapshot/first.pth"; exit 1; }

# 1. extract the zips (kept on disk, skipped if already extracted)
for NAME in NAT2021 NAT2021L; do
    # marker only written after a complete extract, so a cancelled job can't leave a half-extracted set that gets reused
    if [ ! -f $DATA/$NAME/extract_done.txt ]; then
        rm -rf $DATA/$NAME/anno $DATA/$NAME/data_seq
        python -I -c "import zipfile; zipfile.ZipFile('$DATA/NAT2021/test/$NAME.zip').extractall('$DATA')" || exit 1
        touch $DATA/$NAME/extract_done.txt
    fi
    python tools/prepare_nat.py --dataset_root $DATA/$NAME --name $NAME
done

# 2. track + 3. evaluate
cd tools
for NAME in NAT2021 NAT2021L; do
    python test.py --dataset $NAME --snapshot snapshot/first.pth --config ../experiments/config.yaml
    python eval_nat.py --dataset $NAME
done
