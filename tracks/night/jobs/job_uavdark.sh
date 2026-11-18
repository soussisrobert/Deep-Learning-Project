#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=HiFT_uavdark
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Extract UAVDark135-002.rar into training_dataset/UAVDark135/extracted (rar kept), verify the file count against the
# archive listing, then check its frames against NAT2021 / NAT2021L. Guard: stop if home is above 105 GB before extracting.
set -e
export PATH=/apps/anaconda3-2025.06-0/bin:$PATH
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
F=training_dataset/extra_drive/UAVDark135-002.rar
D=training_dataset/UAVDark135/extracted
used=$(du -sm ${HOME} | cut -f1); echo "home before: $used MB"
if [ "$used" -gt 105000 ]; then echo "home above 105 GB, stopping"; exit 2; fi
[ -z "$(ls -A $D 2>/dev/null)" ] || { echo "$D not empty"; exit 3; }
mkdir -p $D
want=$(bsdtar -tf $F | grep -vc '/$'); echo "files in archive listing: $want"
bsdtar -xf $F -C $D
got=$(find $D -type f | wc -l); echo "files extracted: $got"
[ "$got" -eq "$want" ] || { echo "COUNT MISMATCH"; exit 4; }
echo "home after: $(du -sm ${HOME} | cut -f1) MB"
find $D -maxdepth 3 -type d | head -12
python -I tools/leak_one.py UAVDark135 $D
