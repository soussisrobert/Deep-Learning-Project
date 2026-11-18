"""Round 3: choose the JPEG quality from the DEV videos only (pre-registered rule, ROUND3_PLAN.md).

  python scripts/r3_select.py            # needs the stage-1 DEV runs; writes plan/round3/selected.json

Rule (fixed before any round-3 result):
  1. For each quality q in {75, 85, 90, 95}, on the 20 DEV videos:
       clean cost  = mean paired change in success AUC, JPEG-q vs no defence (clean)
       robustness  = success AUC under the selection attack (PGD-10, eps 4, adaptive, surrogate gradient)
  2. Eligible: clean cost >= -0.010 (HiFT's margin over SiamRPN++).
  3. Choose the eligible quality with the highest robustness (ties: the higher quality).
     If none is eligible, choose the quality with the smallest clean cost (ties: the higher quality).
The TEST videos are never read. If the choice is 75, stage-2 runs that are identical to existing
runs are linked instead of re-run (same settings, same videos).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")
D, T = "uav123-dev20_", "uav123-sub40_"
QUALITIES = (75, 85, 90, 95)
CLEAN_COST_MAX = 0.010


def load(n):
    d = RES / n
    if not (d / "summary.json").exists():
        raise SystemExit(f"missing DEV run {n}: finish the stage-1 P1 job first")
    s = json.loads((d / "summary.json").read_text())
    return s, {p["name"]: p["success_auc"] for p in json.loads((d / "per_seq.json").read_text())}


def main():
    dev = json.loads((ROOT / "plan/round3/dev-20.json").read_text())["sequences"]
    base_s, base = load(D + "fp32_clean_nodef")
    if sorted(base) != sorted(dev):
        raise SystemExit("DEV reference does not cover exactly the DEV videos")
    rows = []
    for q in QUALITIES:
        _, c = load(D + f"fp32_clean_jpeg{q}")
        rs, r = load(D + f"fp32_pgd4s10_jpeg{q}_adaptive_diff")
        cost = float(np.mean([c[k] - base[k] for k in dev]))
        rows.append(dict(quality=q, clean_cost=cost, robust_auc=float(np.mean([r[k] for k in dev])),
                         eligible=cost >= -CLEAN_COST_MAX))
    elig = [r for r in rows if r["eligible"]]
    if elig:
        pick = max(elig, key=lambda r: (round(r["robust_auc"], 6), r["quality"]))
        why = "highest robustness among qualities with clean cost >= -0.010"
    else:
        pick = max(rows, key=lambda r: (round(r["clean_cost"], 6), r["quality"]))
        why = "no quality met the clean-cost limit on DEV; chose the smallest clean cost"
    q = pick["quality"]
    out = dict(quality=q, jpeg=f"jpeg{q}", tjpeg=f"tjpeg{q}", reason=why, table=rows,
               chosen_at=time.strftime("%Y-%m-%d %H:%M:%S"), dev_videos=len(dev))
    (ROOT / "plan/round3/selected.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"q{r['quality']}: DEV clean cost {r['clean_cost']:+.4f}  robust AUC {r['robust_auc']:.4f}  "
              f"{'eligible' if r['eligible'] else 'too costly'}")
    print(f"SELECTED jpeg{q} ({why}) -> plan/round3/selected.json")
    if q == 75:   # identical settings already run: link instead of re-running
        same = {"fp32_clean_jpegsel": "fp32_clean_jpeg75",
                "fp32_pgd4s10_jpegsel_adaptive": "fp32_pgd4s10_jpeg75_adaptive",
                "fp32_pgd4s10_jpegsel_adaptive_diff": "fp32_pgd4s10_jpeg75_adaptive_diff",
                "fp32_pgd4s50_jpegsel_adaptive_diff": "fp32_pgd4s50_jpeg75_adaptive_diff"}
        for new, old in same.items():
            src, dst = RES / (T + old), RES / (T + new)
            if (src / "summary.json").exists() and not dst.exists():
                os.symlink(src.name, dst)
                print(f"linked {dst.name} -> {src.name}")


if __name__ == "__main__":
    main()
