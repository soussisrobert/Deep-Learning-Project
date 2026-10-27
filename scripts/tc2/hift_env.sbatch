#!/bin/bash
# HiFT conda env at ~/hift/env, built INSIDE a job (no code runs on the head node). Pinned to the local
# lock (Python 3.13, torch 2.10.0, numpy 2.4.4, OpenCV 4.13.0.92 headless, Pillow 12.1.1, PyYAML 6.0.3, yacs 0.1.8,
# pytest 8.4.2). Deliberate difference from local: torch CUDA 12.6 build (local cu128). Package caches stay
# under ~/hift and are removed afterwards. Writes ~/hift/env/.ready only after a GPU + import check passes.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/msai/ning011/hift/logs/%x_%j.out
#SBATCH --error=/home/msai/ning011/hift/logs/%x_%j.out
set -eo pipefail   # no -u: the cluster's conda activate.d scripts reference unset variables
B=$HOME/hift
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if [ -f "$B/env/.ready" ]; then echo "env already ready"; exit 0; fi
[ -e "$B/env" ] && { echo "partial env at $B/env exists; refusing to overwrite (inspect by hand)"; exit 3; }
export CONDA_PKGS_DIRS=$B/.condapkgs PIP_CACHE_DIR=$B/.pipcache
set +e   # cluster conda hooks may return non-zero internally; checked explicitly below
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
set -e
conda create -y -p "$B/env" -c conda-forge --override-channels python=3.13 pip
set +e; conda activate "$B/env"; set -e
[ "${CONDA_PREFIX:-}" = "$B/env" ] || { echo "env activation failed"; exit 4; }
python -m pip install --no-input torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-input numpy==2.4.4 opencv-python-headless==4.13.0.92 pillow==12.1.1 PyYAML==6.0.3 yacs==0.1.8 pytest==8.4.2
python - <<'EOF'
import cv2, numpy, PIL, torch, yaml, yacs
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "| numpy", numpy.__version__,
      "| cv2", cv2.__version__, "| PIL", PIL.__version__)
EOF
python -m pip freeze > "$B/env/FREEZE.txt"
conda clean -y --all >/dev/null 2>&1 || true
rm -rf "$B/.condapkgs" "$B/.pipcache"
touch "$B/env/.ready"
du -sh "$B/env"
echo "ENV READY $(date --iso-8601=seconds)"
