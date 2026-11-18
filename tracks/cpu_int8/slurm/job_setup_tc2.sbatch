#!/bin/bash
# One-time environment setup on TC2. Submit from the LOGIN node, inside ~/hift-stress:
#   mkdir -p runs && sbatch slurm/job_setup_tc2.sbatch
#SBATCH --job-name=hift_setup
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=runs/setup_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PIN=992aa7c1605360d607af84cb804b93cd8ee81caf
set +u
module load anaconda
eval "$(conda shell.bash hook)"
set -u
env_dir="$HOME/hift-stress/env"
if [[ ! -x "$env_dir/bin/python" ]]; then
  if [[ -e "$env_dir" ]]; then echo "Incomplete environment at $env_dir. Inspect before retrying."; exit 1; fi
  conda create -y -p "$env_dir" --override-channels -c conda-forge python=3.11 pip
fi
set +u; conda activate "$env_dir"; set -u
python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -r requirements.txt

if [[ ! -d third_party/HiFT/pysot ]]; then
  mkdir -p third_party
  git clone https://github.com/vision4robotics/HiFT.git third_party/HiFT \
    || { echo "git clone failed: copy third_party/HiFT from your Mac with rsync and resubmit"; exit 1; }
  git -C third_party/HiFT checkout "$PIN"
fi
python tools/patch_hift.py third_party/HiFT
mkdir -p weights data results
[[ -s weights/general_model.pth ]] || gdown 1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA -O weights/general_model.pth \
  || echo "gdown failed: scp weights/general_model.pth from your Mac"
python -m pip freeze > runs/environment_tc2.txt
python -c 'import torch, cv2, yacs; print("torch", torch.__version__, "cv2", cv2.__version__)'
echo "setup done"
