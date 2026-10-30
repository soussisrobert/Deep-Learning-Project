"""Round-4 independent analysis: accuracy of the selected int8 recipe, per-video failures, speed, JPEG-85, credibility.
Reads saved results only (no project code)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SRC = R.parent / "source"
S40 = sorted(json.loads((SRC / "attack-subset-40.json").read_text())["sequences"])
DEV = sorted(json.loads((SRC / "plan/round3/dev-20.json").read_text())["sequences"])
T, D = "uav123-sub40_", "uav123-dev20_"
J, problems = {}, []
def S(n): return json.loads((R/n/"summary.json").read_text())
def P(n): return {p["name"]: p for p in json.loads((R/n/"per_seq.json").read_text())}
def auc(n, keys, key="success_auc"): p = P(n); return np.array([p[k][key] for k in keys])
def boot(d, n=10000, seed=0):
    d = np.asarray(d, float); bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), median=float(np.median(d)),
                better=int((d > .05).sum()), worse=int((d < -.05).sum()), p_below_m010=float((bs < -.010).mean()))
f = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (median {p['median']:+.4f}; +{p['better']}/-{p['worse']} videos beyond ±0.05)"

# ---------------- validation of round-4 runs
r4 = [n for n in sorted(x.name for x in R.iterdir() if (x/"summary.json").exists()) if ("int8sel" in n or "jpeg85" in n or
      (n.startswith(D) and ("int8" in n or n.endswith("_t4"))))]
for n in r4:
    s, p = S(n), P(n); keys = sorted(p); exp, fr = (DEV, 16826) if n.startswith(D) else (S40, 37681)
    if keys != exp: problems.append(f"{n}: videos")
    if sum(x["frames"] for x in p.values()) != fr: problems.append(f"{n}: frames")
    if len(list((R/n/"preds").glob("*.txt"))) != len(exp): problems.append(f"{n}: preds")
    if abs(np.mean([x["success_auc"] for x in p.values()]) - s["success_auc"]) > 1e-9: problems.append(f"{n}: mean")
    vi = s.get("variant_info", {})
    if "int8sel" in n:
        if vi.get("variant") != "int8bb-s1" or vi.get("backbone_layers_fp32") != ["layer1"] or vi.get("calib_frames") != 4 \
           or not vi.get("calib_spread") or len(vi.get("calibration_sequences", [])) != 40: problems.append(f"{n}: recipe {vi}")
        if set(vi["calibration_sequences"]) & set(S40): problems.append(f"{n}: calibration overlaps TEST")
        if vi.get("calib_set") != int(n[-1]): problems.append(f"{n}: draw")
    if n.startswith(D) and "int8" in n and set(vi.get("calibration_sequences", [])) & set(S40): problems.append(f"{n}: DEV calib uses TEST")
print(f"validated {len(r4)} round-4 runs; problems: {problems or 'none'}"); J["problems"] = problems
cal = [set(S(T+f"int8sel_clean_nodef_t4_cal{k}")["variant_info"]["calibration_sequences"]) for k in (1, 2, 3)]
print("calibration overlap between draws:", len(cal[0] & cal[1]), len(cal[0] & cal[2]), len(cal[1] & cal[2]), "| with DEV:", [len(c & set(DEV)) for c in cal])

# ---------------- 1. accuracy of the selected recipe on TEST
cpu = auc("uav123_fp32_clean_nodef_t4", S40); gpu = auc(T+"fp32_clean_nodef", S40)
draws = np.stack([auc(T+f"int8sel_clean_nodef_t4_cal{k}", S40) for k in (1, 2, 3)])
J["acc"] = {"fp32_cpu_40": float(cpu.mean()), "fp32_gpu_40": float(gpu.mean())}
for k in range(3):
    J["acc"][f"draw{k+1}"] = dict(auc=float(draws[k].mean()), change=boot(draws[k] - cpu))
    print(f"draw {k+1}: AUC {draws[k].mean():.4f}  change {f(J['acc'][f'draw{k+1}']['change'])}")
ch = draws.mean(1) - cpu.mean()
J["acc"]["mean_of_draws"] = float(ch.mean()); J["acc"]["sd_draws"] = float(ch.std(ddof=1))
tcrit = 4.303  # t(0.975, df=2)
J["acc"]["t_interval_over_draws"] = [float(ch.mean() - tcrit * ch.std(ddof=1) / np.sqrt(3)), float(ch.mean() + tcrit * ch.std(ddof=1) / np.sqrt(3))]
avgd = draws.mean(0) - cpu
J["acc"]["draw_averaged_per_video"] = boot(avgd)
# two-level bootstrap: resample videos AND draws
rng = np.random.default_rng(1); bs = []
for _ in range(10000):
    v = rng.integers(0, 40, 40); d = rng.integers(0, 3, 3)
    bs.append((draws[d][:, v].mean() - cpu[v].mean()))
bs = np.array(bs); J["acc"]["two_level"] = dict(lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_below_m010=float((bs < -.010).mean()))
print("mean change over draws", ch.mean(), "SD", ch.std(ddof=1), "t-interval", J["acc"]["t_interval_over_draws"])
print("draw-averaged per-video", f(J["acc"]["draw_averaged_per_video"]), "| two-level 95%", J["acc"]["two_level"])
# official precision
_rs = {x["name"]: x for x in json.loads((R/"uav123_fp32_clean_nodef_t4"/"rescore.json").read_text())["per_seq"]}  # official rule, re-scored from saved boxes
pcpu = np.array([_rs[k]["precision_20_official"] for k in S40])
pdr = np.stack([auc(T+f"int8sel_clean_nodef_t4_cal{k}", S40, "precision_20_official") for k in (1, 2, 3)])
J["acc"]["precision_official"] = dict(fp32_cpu=float(pcpu.mean()), draws=[float(x) for x in pdr.mean(1)], change=boot(pdr.mean(0) - pcpu))
print("official precision@20: fp32 CPU", pcpu.mean(), "draws", pdr.mean(1), "change", f(J["acc"]["precision_official"]["change"]))

# ---------------- noise floor: how much do 'harmless' changes move the 40-video score?
nf = {}
nf["CPU vs GPU fp32 (same weights, different hardware arithmetic)"] = boot(cpu - gpu)
nf["random noise eps=1 (GPU)"] = boot(auc(T+"fp32_noise1_nodef", S40) - gpu)
old = [auc(T+"int8bb_clean_nodef_t4", S40)] + [auc(T+f"int8bb_clean_nodef_t4_cal{k}", S40) for k in (1, 2, 3)]
for i, o in enumerate(old): nf[f"old int8bb recipe draw {i}"] = boot(o - cpu)
nf["int8sel draw 1 vs draw 2 (same recipe, different calibration videos)"] = boot(draws[0] - draws[1])
nf["int8sel draw 1 vs draw 3"] = boot(draws[0] - draws[2])
nf["int8sel draw 2 vs draw 3"] = boot(draws[1] - draws[2])
J["noise_floor"] = nf
for k, x in nf.items(): print(f"NOISE {k}: {f(x)}")
old_ch = [float((o - cpu).mean()) for o in old]
J["old_recipe"] = dict(changes=old_ch, mean=float(np.mean(old_ch)), sd=float(np.std(old_ch, ddof=1)))
print("old recipe 4 draws", old_ch, "mean", np.mean(old_ch), "SD", np.std(old_ch, ddof=1))

# ---------------- 2. where accuracy is lost (per video)
init = {}
for k in S40:
    b = np.loadtxt(R/"uav123_fp32_clean_nodef_t4"/"preds"/f"{k}.txt", delimiter=",", ndmin=2)
    init[k] = b[0]  # first row = ground-truth initial box (tracker is initialised on it)
pref = {**P("uav123_fp32_clean_nodef_t4"), **{k: {**P(T+"int8sel_clean_nodef_t4_cal1")[k]} for k in S40}}  # absent-frame counts come from ground truth (same in every run)
rows = []
for i, k in enumerate(S40):
    pr = np.loadtxt(R/"uav123_fp32_clean_nodef_t4"/"preds"/f"{k}.txt", delimiter=",", ndmin=2)
    c = pr[:, :2] + pr[:, 2:] / 2; diag = np.sqrt(pr[:, 2] ** 2 + pr[:, 3] ** 2)
    motion = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1) / diag[1:]))  # prediction-derived proxy
    rows.append(dict(video=k, category="".join(ch for ch in k.split("_")[0] if ch.isalpha()), frames=pref[k]["frames"],
                     init_size=float(np.sqrt(init[k][2] * init[k][3])), absent=pref[k]["absent_frames"],
                     absent_frac=pref[k]["absent_frames"] / pref[k]["frames"], motion_proxy=motion,
                     fp32_cpu=float(cpu[i]), fp32_gpu=float(gpu[i]), d1=float(draws[0, i] - cpu[i]), d2=float(draws[1, i] - cpu[i]),
                     d3=float(draws[2, i] - cpu[i]), davg=float(avgd[i]), cpu_vs_gpu=float(cpu[i] - gpu[i]),
                     jpeg75=float(auc(T+"fp32_clean_jpeg75", [k])[0] - gpu[i]), jpeg85=float(auc(T+"fp32_clean_jpeg85", [k])[0] - gpu[i])))
rows.sort(key=lambda r: r["davg"])
J["per_video"] = rows
print("\nWORST videos for int8sel (draw-averaged change vs fp32 CPU):")
for r in rows[:6]:
    print(f"  {r['video']:11s} fp32 {r['fp32_cpu']:.3f}  draws {r['d1']:+.3f} {r['d2']:+.3f} {r['d3']:+.3f}  avg {r['davg']:+.3f} | init size {r['init_size']:.0f}px absent {r['absent_frac']:.0%} motion {r['motion_proxy']:.3f} | cpu-gpu {r['cpu_vs_gpu']:+.3f} jpeg75 {r['jpeg75']:+.3f} jpeg85 {r['jpeg85']:+.3f}")
print("BEST:")
for r in rows[-3:]:
    print(f"  {r['video']:11s} fp32 {r['fp32_cpu']:.3f}  avg {r['davg']:+.3f}")
dv = np.array([r["davg"] for r in rows])
J["concentration"] = dict(within_002=int((abs(dv) <= .02).sum()), lost_gt_005=int((dv < -.05).sum()),
                          share_of_loss_top2=float(np.sort(dv)[:2].sum() / dv.sum()) if dv.sum() < 0 else None,
                          mean_without_worst2=float(np.sort(dv)[2:].mean()))
print("concentration:", J["concentration"])
# any-draw large failures
anyfail = [r["video"] for r in rows if min(r["d1"], r["d2"], r["d3"]) < -.05]
J["any_draw_failure_videos"] = anyfail; print("videos with a >0.05 loss in at least one draw:", anyfail)
# correlations with proxies (Spearman), descriptive
from itertools import combinations
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
for key in ("init_size", "absent_frac", "motion_proxy", "fp32_cpu", "frames"):
    J.setdefault("spearman_abs_change", {})[key] = spearman([r[key] for r in rows], [abs(r["davg"]) for r in rows])
print("Spearman |change| vs proxies:", {k: round(v, 2) for k, v in J["spearman_abs_change"].items()})
# small-object split (median), descriptive
med = np.median([r["init_size"] for r in rows])
small = [r["davg"] for r in rows if r["init_size"] <= med]; big = [r["davg"] for r in rows if r["init_size"] > med]
J["size_split"] = dict(median_px=float(med), small=boot(small), large=boot(big))
print("size split (init box sqrt-area <= median):", f(J["size_split"]["small"]), "|", f(J["size_split"]["large"]))
# same fragile videos across perturbations?
fr = {"int8sel": [r["video"] for r in rows if r["davg"] < -.05], "jpeg75": [r["video"] for r in rows if r["jpeg75"] < -.05],
      "jpeg85": [r["video"] for r in rows if r["jpeg85"] < -.05], "cpu_vs_gpu": [r["video"] for r in rows if abs(r["cpu_vs_gpu"]) > .05]}
J["fragile"] = fr; print("videos moving >0.05:", fr)

# ---------------- 3. speed
E = {f.stem: json.loads(f.read_text()) for f in (R/"edge_round4").glob("*.json")}
sp = {}
for t in (1, 4):
    f32 = E[f"cpu_t{t}_fp32"]
    for v in ("int8bb", "int8bb-s1"):
        e = E[f"cpu_t{t}_{v}"]
        sp[f"t{t}_{v}"] = dict(net_ms=e["latency_ms_mean"], fp32_ms=f32["latency_ms_mean"], speedup=f32["latency_ms_mean"] / e["latency_ms_mean"],
                               backbone_ms=e["backbone_ms_mean"], head_ms=e["head_ms_mean"], fp32_backbone=f32["backbone_ms_mean"],
                               fp32_head=f32["head_ms_mean"], size_mb=e["size_mb"], fp32_size=f32["size_mb"], same_peak=e["same_peak_fraction"])
J["speed_network"] = sp
for k, x in sp.items(): print("NET", k, {kk: round(vv, 3) for kk, vv in x.items()})
# tracking throughput on matched videos (frame-weighted, per-sequence timing; image loading excluded)
def track_ms(n, keys):
    p = P(n); fr_ = np.array([p[k]["frames"] - 1 for k in keys]); fps = np.array([p[k]["fps"] for k in keys])
    return float((fr_ / fps).sum() / fr_.sum() * 1e3)
tr = {"fp32_cpu_on_40 (round 1 job)": track_ms("uav123_fp32_clean_nodef_t4", S40)}
for k in (1, 2, 3): tr[f"int8sel draw {k}"] = track_ms(T+f"int8sel_clean_nodef_t4_cal{k}", S40)
tr["old int8bb (round 2 job)"] = track_ms(T+"int8bb_clean_nodef_t4", S40)
J["speed_tracking_ms_per_frame"] = tr
print("tracking ms/frame (per-frame tracker step, no image loading):", {k: round(v, 1) for k, v in tr.items()},
      "-> speed-up", round(tr["fp32_cpu_on_40 (round 1 job)"] / np.mean([tr[f'int8sel draw {k}'] for k in (1, 2, 3)]), 2))
J["speed_whole_job_min"] = {n: S(n)["total_minutes"] for n in [T+f"int8sel_clean_nodef_t4_cal{k}" for k in (1, 2, 3)]}
s1 = S(T+"int8sel_clean_nodef_t4_cal1")["ms_per_frame"]; f1 = S("uav123_fp32_clean_nodef_t4")["ms_per_frame"]
J["speed_components"] = dict(int8sel=s1, fp32_123=f1)
print("ms_per_frame components int8sel", {k: round(v, 1) for k, v in s1.items()}, "fp32(123)", {k: round(v, 1) for k, v in f1.items()})

# ---------------- 4. JPEG and DEV-vs-TEST
dev_gpu = auc(D+"fp32_clean_nodef", DEV)
jd = {q: boot(auc(D+f"fp32_clean_jpeg{q}", DEV) - dev_gpu) for q in (75, 85, 90, 95)}
jt = {75: boot(auc(T+"fp32_clean_jpeg75", S40) - gpu), 85: boot(auc(T+"fp32_clean_jpeg85", S40) - gpu)}
J["jpeg_dev_vs_test"] = dict(dev=jd, test=jt)
for q in (75, 85): print(f"JPEG{q}: DEV {f(jd[q])} | TEST {f(jt[q])}")
# DEV int8 vs TEST: does DEV predict?
dev_cpu = auc(D+"fp32_clean_nodef_t4", DEV)
devrec = {}
for tag in ("int8bb_c20x2", "int8bb_c40x4s", "int8bb-s1_c20x2", "int8bb-s1_c40x4s"):
    m = np.stack([auc(D+f"{tag}_cal{k}", DEV) for k in (1, 2, 3)])
    devrec[tag] = dict(mean=float((m.mean(1) - dev_cpu.mean()).mean()), per_video=boot(m.mean(0) - dev_cpu))
J["dev_int8"] = devrec
print("DEV recipes (draw-averaged per-video):", {k: f(v["per_video"]) for k, v in devrec.items()})
# DEV fp32: CPU vs GPU (is the DEV reference itself odd?)
J["dev_cpu_vs_gpu"] = boot(dev_cpu - dev_gpu); print("DEV fp32 CPU vs GPU:", f(J["dev_cpu_vs_gpu"]))
dpv = []
m = np.stack([auc(D+f"int8bb-s1_c40x4s_cal{k}", DEV) for k in (1, 2, 3)]).mean(0)
for i, k in enumerate(DEV): dpv.append((k, float(m[i] - dev_cpu[i]), float(dev_cpu[i]), float(dev_gpu[i])))
dpv.sort(key=lambda x: -abs(x[1])); J["dev_selected_top"] = dpv[:4]
print("DEV selected recipe, largest per-video moves:", [(a, round(b, 3), round(c, 3), round(d, 3)) for a, b, c, d in dpv[:4]])
json.dump(J, open(OUT/"numbers4.json", "w"), indent=1, default=float)

# ================= figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, GR = "#2a6fb0", "#d0703c", "#3b9a6b", "#8c8c8c"
# fig8: per-video change, sorted, with draws
fig, ax = plt.subplots(figsize=(7.4, 3.8))
x = np.arange(40)
for j, col in zip(("d1", "d2", "d3"), ("#9cc3e6", "#5b9bd5", "#1f4e79")):
    ax.scatter(x, [r[j] for r in rows], s=10, color=col, label=f"draw {j[1]}")
ax.plot(x, [r["davg"] for r in rows], color="k", lw=1, label="mean of 3 draws")
ax.axhline(0, color=GR, lw=.6)
for i, r in enumerate(rows[:3]): ax.annotate(r["video"], (i, r["davg"]), xytext=(4, -10), textcoords="offset points", fontsize=8)
ax.set_xticks([]); ax.set_xlabel("40 TEST videos, sorted by mean change"); ax.set_ylabel("change in success AUC vs fp32 CPU")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right"); ax.set_title("int8bb-s1: most videos unchanged, a few tip over", loc="left")
fig.tight_layout(); fig.savefig(OUT/"fig8_int8sel_per_video.png", dpi=160); plt.close(fig)
# fig9: noise floor comparison
fig, ax = plt.subplots(figsize=(7.2, 3.9))
items = [("int8bb-s1 draw 1", J["acc"]["draw1"]["change"]), ("int8bb-s1 draw 2", J["acc"]["draw2"]["change"]), ("int8bb-s1 draw 3", J["acc"]["draw3"]["change"])] + \
        [(f"old int8bb draw {i}", nf[f'old int8bb recipe draw {i}']) for i in range(4)] + \
        [("fp32: CPU vs GPU", nf["CPU vs GPU fp32 (same weights, different hardware arithmetic)"]), ("noise ε=1 (GPU)", nf["random noise eps=1 (GPU)"]),
         ("JPEG-75 (GPU)", jt[75]), ("JPEG-85 (GPU)", jt[85])]
for i, (lab, b) in enumerate(items):
    col = C1 if "s1" in lab else (C3 if "old" in lab else (C2 if "JPEG" in lab else GR))
    ax.errorbar(b["mean"], i, xerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]], fmt="o", color=col, capsize=3)
ax.axvline(0, color="k", lw=.6); ax.axvline(-.010, color=C2, ls="--", lw=1); ax.text(-.0105, -.6, "−0.010 rule", color=C2, fontsize=8, ha="right")
ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l, _ in items], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("change in success AUC on the 40 TEST videos (95% bootstrap over videos)")
ax.set_title("New int8 recipe vs old recipe and reference changes (TEST, 40 videos)", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"fig9_noise_floor.png", dpi=160); plt.close(fig)
# fig10: speed definitions
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labs = ["network only\n(4 threads, same job)", "tracker step\n(matched 40 videos)"]
a = [sp["t4_int8bb-s1"]["fp32_ms"], tr["fp32_cpu_on_40 (round 1 job)"]]; b = [sp["t4_int8bb-s1"]["net_ms"], np.mean([tr[f"int8sel draw {k}"] for k in (1, 2, 3)])]
xx = np.arange(2); w = .35
ax.bar(xx - w/2, a, w, color=GR, label="fp32"); ax.bar(xx + w/2, b, w, color=C1, label="int8bb-s1")
for i in range(2): ax.text(xx[i] + w/2, b[i] + 2, f"{a[i]/b[i]:.2f}× faster", ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_ylabel("milliseconds per frame (Xeon CPU)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Speed-up holds at the tracker level (server CPU)", loc="left"); fig.tight_layout(); fig.savefig(OUT/"fig10_speed.png", dpi=160); plt.close(fig)
print("figures written")
