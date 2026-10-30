"""Turn an unpacked UAV123 download into the layout the test scripts read.

Run from ~/hift-stress after the data job (works with either data job):
    python scripts/convert_uav123.py
It finds the folder that contains anno/UAV123 anywhere under data/, then writes
    data/uav123_flat   (123 sequences, the main benchmark)
    data/uav20l_flat   (20 long sequences, optional)
    data/uav123_mini.tar   (5 sequences x 60 frames to copy to a laptop)
and prints how many sequences and frames each holds. Safe to run again.
"""
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hiftstress.data import load_dataset  # noqa: E402


def find_raw(data_dir):
    hits = sorted(p.parent.parent for p in data_dir.rglob("anno/UAV123")
                  if p.is_dir() and "_flat" not in str(p) and "mini" not in str(p))
    if not hits:
        sys.exit(f"No anno/UAV123 folder found under {data_dir}. Has the archive finished unpacking?")
    return hits[0]


def main():
    data = ROOT / "data"
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else find_raw(data)
    print(f"raw UAV123 root: {raw}")
    prep = [sys.executable, str(ROOT / "tools" / "prepare_uav.py"), "--raw", str(raw)]
    subprocess.run(prep + ["--set", "UAV123", "--out", str(data / "uav123_flat")], check=True)
    subprocess.run(prep + ["--set", "UAV20L", "--out", str(data / "uav20l_flat")], check=False)
    for name in ("uav123_flat", "uav20l_flat"):
        try:
            seqs = load_dataset("flat", str(data / name))
            print(f"data/{name}: {len(seqs)} sequences, {sum(len(s) for s in seqs)} frames")
        except SystemExit as e:
            print(f"data/{name}: {e}")
    seqs = load_dataset("flat", str(data / "uav123_flat"), max_seqs=5)
    with tarfile.open(data / "uav123_mini.tar", "w", dereference=True) as tar:
        for s in seqs:
            for f in s.frames[:60]:
                tar.add(str(f), arcname=f"uav123_mini/data_seq/{s.name}/{Path(f).name}")
            tmp = data / f"_{s.name}.txt"
            tmp.write_text("\n".join(",".join(f"{v:g}" for v in row) for row in s.gt[:60]) + "\n")
            tar.add(str(tmp), arcname=f"uav123_mini/anno/{s.name}.txt")
            tmp.unlink()
    print(f"wrote data/uav123_mini.tar: {[s.name for s in seqs]}")
    print("Expect 123 sequences / 112578 frames for uav123_flat. Next: sbatch slurm/job_smoke_tc2.sbatch")


if __name__ == "__main__":
    main()
