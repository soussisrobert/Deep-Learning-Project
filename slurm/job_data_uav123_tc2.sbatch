#!/bin/bash
# Download UAV123 (full 30 fps release, 14 GB) straight onto TC2 and convert it.
# Needs no GPU. Submit from the login node inside ~/hift-stress:
#   sbatch slurm/job_data_uav123_tc2.sbatch
# Source: Hugging Face mirror xche32/UAV123 (a repack of the KAUST release; the official
# KAUST page no longer resolves). Resubmitting resumes a half-finished download.
# If the download says 401/gated: on the login node run  hf auth login  (paste a read token
# from huggingface.co/settings/tokens; accept the dataset terms on its page if asked), then resubmit.
# Needs about 30 GB free while extracting (archive + images); the archive is deleted after.
#SBATCH --job-name=hift_data
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=runs/data_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/hift-stress/env/bin:$PATH"
python -c "import huggingface_hub" 2>/dev/null || python -m pip install -q huggingface_hub
mkdir -p data/raw
df -h data | tail -1
if [[ ! -d data/raw/UAV123_extracted_ok ]]; then
  echo "[$(date +%T)] downloading with huggingface_hub (resumable; uses your token if you ran 'hf auth login')"
  python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="xche32/UAV123", repo_type="dataset",
                       filename="UAV123.tar.gz", local_dir="data/raw")
print("downloaded", path)
PY
  ls -lh data/raw/UAV123.tar.gz
  echo "[$(date +%T)] checking archive"
  gzip -t data/raw/UAV123.tar.gz
  echo "[$(date +%T)] extracting"
  tar -xzf data/raw/UAV123.tar.gz -C data/raw
  mkdir -p data/raw/UAV123_extracted_ok && rm -f data/raw/UAV123.tar.gz
fi
# the archive may unpack as data/raw/UAV123/{data_seq,anno} or straight into data/raw
RAW=$(dirname "$(find data/raw -maxdepth 3 -type d -path '*/anno/UAV123' | head -1)")
RAW=$(dirname "$RAW")
echo "raw root: $RAW"; ls "$RAW"
python tools/prepare_uav.py --raw "$RAW" --set UAV123 --out data/uav123_flat
python tools/prepare_uav.py --raw "$RAW" --set UAV20L --out data/uav20l_flat || true
python - <<'PY'
import sys; sys.path.insert(0, ".")
from hiftstress.data import load_dataset
for root in ("data/uav123_flat", "data/uav20l_flat"):
    try:
        s = load_dataset("flat", root)
        print(f"{root}: {len(s)} sequences, {sum(len(x) for x in s)} frames")
    except SystemExit as e:
        print(f"{root}: {e}")
PY
# small subset to copy back to a laptop for local quick runs (real files, ~5 sequences x 60 frames)
python - <<'PY'
import sys, tarfile; sys.path.insert(0, ".")
from hiftstress.data import load_dataset
seqs = load_dataset("flat", "data/uav123_flat", max_seqs=5)
with tarfile.open("data/uav123_mini.tar", "w", dereference=True) as tar:
    for s in seqs:
        for f in s.frames[:60]:
            tar.add(str(f), arcname=f"uav123_mini/data_seq/{s.name}/{f.name}")
        lines = "\n".join(",".join(f"{v:g}" for v in row) for row in s.gt[:60]) + "\n"
        p = f"data/_{s.name}.txt"; open(p, "w").write(lines)
        tar.add(p, arcname=f"uav123_mini/anno/{s.name}.txt")
print("wrote data/uav123_mini.tar:", [s.name for s in seqs])
PY
rm -f data/_*.txt
echo "done"
