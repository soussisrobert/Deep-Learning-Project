"""Copy the night track's per-video files (already in the common format) into results/night/."""
import pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "tracks/night/evidence"
for f in sorted(SRC.glob("per_video_*.csv")):
    shutil.copy2(f, ROOT / "results/night" / f.name)
    print("copied", f.name)
