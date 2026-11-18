#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --job-name=HiFT_uavdark_chk
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Read-only: why does the rar listing (125,875 non-directory lines) differ from the extracted file count (125,736)?
export PATH=/apps/anaconda3-2025.06-0/bin:$PATH
cd ${HIFT_ROOT}
F=training_dataset/extra_drive/UAVDark135-002.rar
T=$(mktemp)
bsdtar -tf $F > $T
echo "listing lines: $(wc -l < $T)  ending in slash: $(grep -c '/$' $T)  unique names: $(sort -u $T | wc -l)"
echo "listed jpg: $(grep -ci '\.jpg$' $T)  txt: $(grep -ci '\.txt$' $T)  other:"; grep -vi -E '\.(jpg|txt)$|/$' $T | head -20
echo "duplicated names (first 10):"; sort $T | uniq -d | head -10
echo "listed jpg unique: $(grep -i '\.jpg$' $T | sort -u | wc -l)"
D=training_dataset/UAVDark135/extracted
echo "extracted jpg: $(find $D -type f -iname '*.jpg' | wc -l)  txt: $(find $D -type f -iname '*.txt' | wc -l)"
echo "listed dirs without slash? zero-length entries are not checked here"
rm -f $T
