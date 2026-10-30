# Sourced by every TC2 job: activate the private env and log the hardware.
export PATH="$HOME/hift-stress/env/bin:$PATH"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
test -x "$HOME/hift-stress/env/bin/python" || { echo "env missing: run slurm/job_setup_tc2.sbatch"; exit 1; }
test -s weights/general_model.pth || { echo "weights/general_model.pth missing"; exit 1; }
# Benchmark: UAV123 (30 fps) by default; DATASET=dtb70 ROOT=data/DTB70 TAG=dtb70 if you have DTB70.
export DATASET=${DATASET:-flat} ROOT=${ROOT:-data/uav123_flat} TAG=${TAG:-uav123}
test -d "$ROOT" || { echo "$ROOT missing: run slurm/job_data_uav123_tc2.sbatch first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no GPU visible"
lscpu | grep -E "Model name|^CPU\(s\)" || true
python -c 'import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")'
