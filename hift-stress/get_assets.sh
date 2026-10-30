#!/bin/bash
# Download the official HiFT weights, and check the DTB70 benchmark is in place.
#   bash get_assets.sh
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && source .venv/bin/activate
mkdir -p weights data

# Official HiFT weights (link from the HiFT README, Google Drive)
if [[ ! -s weights/general_model.pth ]]; then
  echo "Downloading HiFT general_model.pth ..."
  if ! gdown 1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA -O weights/general_model.pth; then
    echo "gdown failed. Download it in a browser instead:"
    echo "  https://drive.google.com/file/d/1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA/view"
    echo "and save it as $(pwd)/weights/general_model.pth"
  fi
fi
ls -lh weights/general_model.pth 2>/dev/null || true

# DTB70: 70 drone videos, ~16k frames, the benchmark HiFT reports.
if [[ -d data/DTB70 ]]; then
  n=$(find data/DTB70 -maxdepth 2 -name groundtruth_rect.txt | wc -l | tr -d ' ')
  echo "DTB70 found: $n sequences (expect 70)"
else
  cat <<'MSG'
DTB70 not found. The authors host the DATASET on Baidu (the repo's MDNet/SODLT
weights are for other trackers; ignore them):
  https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ   (linked from github.com/flyers/drone-tracking)
If Baidu is not workable, use UAV123 (default): slurm/job_data_uav123_tc2.sbatch downloads it on TC2.
Unzip so that the layout is   data/DTB70/<sequence>/img/00001.jpg   and
                              data/DTB70/<sequence>/groundtruth_rect.txt
MSG
fi
cat <<'MSG'

Optional second benchmark, UAV123@10fps (~4.6 GB):
  https://cemse.kaust.edu.sa/ivul/uav123  -> Dataset_UAV123_10fps.zip (+ the tracker benchmark zip for configSeqs*.m)
  python tools/prepare_uav.py --raw data/UAV123_10fps --set UAV123_10fps --out data/uav10fps_flat --config <configSeqs.m>
  then use  --dataset flat --root data/uav10fps_flat
MSG
