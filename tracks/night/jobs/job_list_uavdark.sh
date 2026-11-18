#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=HiFT_list_uavdark
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# List the UAVDark135 rar (read-only): is it a complete archive or one volume of a multi-part set? Nothing is extracted.
export PATH=/apps/anaconda3-2025.06-0/bin:$PATH
cd ${HIFT_ROOT}
F=training_dataset/extra_drive/UAVDark135-002.rar
head -c 16 $F | od -c | head -2
bsdtar --version
bsdtar -tf $F 2>&1 | head -30
echo "entries: $(bsdtar -tf $F 2>/dev/null | wc -l)"
