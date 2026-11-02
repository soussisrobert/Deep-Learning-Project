"""One table of every tracking run (all rounds) + per-video success + the overview figure.

  python analysis/all-rounds/make_overview.py <results_dir> [<round1_results> <round2_results> <round3_results> <round4_results>]

<results_dir> is the cumulative results folder from the round-5 handoff (it holds every run).
The optional earlier folders only fill the "round" column (the round in which a run first appeared).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
RETIME = None
if "--retime" in args:                      # matched re-timing summary (4 threads) replaces the round-4 bench speeds
    i = args.index("--retime"); RETIME = json.loads(Path(args[i + 1]).read_text()); del args[i:i + 2]
R = Path(args[0]); OUT = Path(__file__).resolve().parent
earlier = [Path(p) for p in args[1:]]
first = {}
for i, p in enumerate(earlier, 1):
    for s in p.glob("*/summary.json"):
        first.setdefault(s.parent.name, i)

SETS = {"uav123-sub40_": "TEST-40", "uav123-dev20_": "DEV-20", "uav123-nt83_": "nonTEST-83", "uav123_": "ALL-123"}
rows, perv, A = [], {}, {}
for s in sorted(R.glob("*/summary.json")):
    run = s.parent.name
    if "__p" in run:          # parts of split runs; the merged run is listed instead
        continue
    d = json.loads(s.read_text())
    st = next((v for k, v in SETS.items() if run.startswith(k)), "other")
    ms = d.get("ms_per_frame") or {}
    A[run] = d["success_auc"]
    rows.append(dict(run=run, round=first.get(run, 5), video_set=st, videos=d["sequences"], frames=d["frames"],
                     model=d.get("variant", "fp32"), device=d.get("device"), attack=d.get("attack", "none"),
                     eps=d.get("eps", 0), steps=d.get("steps", 0), adaptive=d.get("adaptive", False),
                     attack_gradient=d.get("grad") or "", defence=d.get("defense", "none"),
                     corruption=d.get("corruption", "none"), calib_set=d.get("calib_set", ""),
                     success_auc=round(d["success_auc"], 4),
                     precision20_official=round(d["precision_20_official"], 4) if "precision_20_official" in d else "",
                     precision20_strict=round(d["precision_20"], 4),
                     ms_per_frame=round(ms["total"], 1) if "total" in ms else ""))
    ps = s.parent / "per_seq.json"
    if ps.exists():
        perv[run] = {p["name"]: p["success_auc"] for p in json.loads(ps.read_text())}
with open(OUT / "all_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
vids = sorted({v for d in perv.values() for v in d})
with open(OUT / "per_video_success.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["video"] + list(perv))
    for v in vids:
        w.writerow([v] + [f"{perv[r][v]:.4f}" if v in perv[r] else "" for r in perv])
print(f"{len(rows)} runs, {len(vids)} videos")

# ---- overview figure
T = "uav123-sub40_"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
bars = [("Clean", A[T + "fp32_clean_nodef"], "#8c8c8c"),
        ("Random noise\nε=4", A[T + "fp32_noise4_nodef"], "#8c8c8c"),
        ("PGD-10\nattack ε=4", A[T + "fp32_pgd4s10_nodef"], "#c0392b"),
        ("PGD-50\nattack ε=4", A[T + "fp32_pgd4s50_nodef"], "#c0392b"),
        ("JPEG-75 vs\nPGD-50 that\nknows it", A[T + "fp32_pgd4s50_jpeg75_adaptive_diff"], "#2a6fb0"),
        ("JPEG-75,\nclean", A[T + "fp32_clean_jpeg75"], "#2a6fb0")]
x = np.arange(len(bars))
ax[0].bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=.65)
for i, b in enumerate(bars):
    ax[0].text(i, b[1] + .01, f"{b[1]:.3f}", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels([b[0] for b in bars], fontsize=8)
ax[0].set_ylabel("success AUC (40 TEST videos)"); ax[0].set_ylim(0, .7)
ax[0].set_title("A. An invisible attack breaks HiFT; JPEG gets some back", loc="left", fontsize=10)
E4 = {p.stem: json.loads(p.read_text()) for p in (R / "edge_round4").glob("cpu_t4_*.json")}
if RETIME:
    bt = RETIME["by_threads"]["4"]
    ms_fp32, ms_int8 = bt["fp32_net_ms"]["median"], bt["int8bb-s1_net_ms"]["median"]
    spd = bt["speedup_net_ms"]["median"]
else:
    ms_fp32, ms_int8 = E4["cpu_t4_fp32"]["latency_ms_mean"], E4["cpu_t4_int8bb-s1"]["latency_ms_mean"]
    spd = ms_fp32 / ms_int8
pts = [("fp32 (original)", ms_fp32, A["uav123_fp32_clean_nodef_t4"], "#8c8c8c")]
nt = {k: v for k, v in perv["uav123-nt83_int8sel_clean_nodef_t4_cal0"].items()}
t40 = [perv[T + f"int8sel_clean_nodef_t4_cal{k}"] for k in (1, 2, 3)]
pooled = np.mean([np.mean([d[v] for d in t40]) if v in t40[0] else nt[v] for v in perv["uav123_fp32_clean_nodef_t4"]])
pts.append(("int8 backbone,\nlayer1 kept fp32\n(our pick)", ms_int8, pooled, "#2a6fb0"))
for (n, ms, auc, c), (dx, dy) in zip(pts, [(0, -34), (10, -38)]):
    ax[1].scatter(ms, auc, s=90, color=c, zorder=3)
    ax[1].annotate(f"{n}\n{ms:.0f} ms, AUC {auc:.3f}", (ms, auc), textcoords="offset points", xytext=(dx, dy),
                   fontsize=8, ha="center" if dx == 0 else "left")
ax[1].axhline(0.589, color="k", ls=":", lw=.8); ax[1].text(68, .5893, "HiFT paper 0.589", fontsize=8, va="bottom")
ax[1].text(37, .5735, "Per video: +0.003 overall, but −0.004 if two lucky\nvideos (person19_2, uav4) are left out. Small, real cost.",
           fontsize=8, color="#444")
ax[1].set_xlim(35, 110); ax[1].set_ylim(.57, .60)
ax[1].set_xlabel("network time per frame, 4 CPU threads (ms)"); ax[1].set_ylabel("success AUC (all 123 videos)")
ax[1].set_title(f"B. int8: {spd:.2f}× faster network for a very small accuracy cost", loc="left", fontsize=10)
ax[1].set_xlabel("network time per frame, 4 CPU threads (ms)" + ("\nmatched re-timing, median of 6 sessions" if RETIME else ""), fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig0_overview.png", dpi=160)
print("pooled int8 123:", round(float(pooled), 4))
