"""Convert the CPU track's wide per-video table to the common long format.

Source: tracks/cpu_int8/analysis/all-rounds/per_video_success.csv (one column per run,
success AUC rounded to 4 decimals by make_overview.py).
Output: results/cpu/per_video_UAV123.csv
"""
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
T = ROOT / "tracks/cpu_int8"
SRC = T / "analysis/all-rounds/per_video_success.csv"
FRAMES = json.loads((T / "plan/uav123_frames.json").read_text())["frames"]
# model name, seed (calibration draw), source column
RUNS = [
    ("fp32_cpu", 0, "uav123_fp32_clean_nodef_t4"),
    ("int8bb-s1", 1, "uav123-sub40_int8sel_clean_nodef_t4_cal1"),
    ("int8bb-s1", 2, "uav123-sub40_int8sel_clean_nodef_t4_cal2"),
    ("int8bb-s1", 3, "uav123-sub40_int8sel_clean_nodef_t4_cal3"),
    ("int8bb-s1_nontest", 0, "uav123-nt83_int8sel_clean_nodef_t4_cal0"),
    ("fp32_gpu", 0, "uav123-sub40_fp32_clean_nodef"),
    ("fp32_gpu_pgd4s10", 0, "uav123-sub40_fp32_pgd4s10_nodef"),
]

def main():
    rows = list(csv.DictReader(open(SRC)))
    out = ROOT / "results/cpu/per_video_UAV123.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "video", "frames", "model", "seed", "success_auc", "precision20", "toolkit"])
        for model, seed, col in RUNS:
            for r in rows:
                if r.get(col):
                    w.writerow(["UAV123", r["video"], FRAMES[r["video"]], model, seed, r[col], "", "own"])
    print("wrote", out)

if __name__ == "__main__":
    main()
