"""Round-3 independent analysis: validation, paired statistics and figures (saved results only)."""
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
def L(n):
    s = json.loads((R/n/"summary.json").read_text())
    s["per"] = {p["name"]: p for p in json.loads((R/n/"per_seq.json").read_text())}
    return s
def v(s, keys): return np.array([s["per"][k]["success_auc"] for k in keys])
def boot(d, n=10000, seed=0):
    d = np.asarray(d, float); bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), better=int((d > .05).sum()), worse=int((d < -.05).sum()))
f = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (n={p['n']}, +{p['better']}/-{p['worse']})"
J, problems = {}, []

# ---------- validation of every round-3 run
r3 = [n for n in sorted(p.name for p in R.iterdir() if (p/"summary.json").exists())
      if n.startswith(D) or any(k in n for k in ("pgd4s50", "_diff", "jpegsel", "_cal", "int8bb_clean_jpegsel"))]
for n in r3:
    s = L(n); keys = sorted(s["per"])
    if "__p" in n: continue
    exp, fr = (DEV, 16826) if n.startswith(D) else (S40, 37681)
    if keys != exp: problems.append(f"{n}: video set")
    if sum(p["frames"] for p in s["per"].values()) != fr: problems.append(f"{n}: frames")
    if len(list((R/n/"preds").glob("*.txt"))) != len(exp): problems.append(f"{n}: preds")
    if abs(np.mean(v(s, keys)) - s["success_auc"]) > 1e-9: problems.append(f"{n}: mean")
    if "pgd4s50" in n and s["steps"] != 50: problems.append(f"{n}: steps {s['steps']}")
    if "_diff" in n and s.get("grad") != "diff": problems.append(f"{n}: grad {s.get('grad')}")
    if "jpegsel" in n and s["defense"] != "jpeg75": problems.append(f"{n}: defense {s['defense']}")
    if "_cal" in n and s.get("calib_set") != int(n[-1]): problems.append(f"{n}: calib_set")
print(f"validated {len([n for n in r3 if '__p' not in n])} round-3 runs; problems: {problems or 'none'}")
J["problems"] = problems
# DEV/TEST/calibration separation
cal40 = {}
for n in [T+"int8bb_clean_nodef_t4"] + [T+f"int8bb_clean_nodef_t4_cal{k}" for k in (1,2,3)] + [T+"int8bb_clean_jpegsel_t4"]:
    c = L(n)["variant_info"]["calibration_sequences"]; cal40[n] = c
    print(f"{n}: calib {len(c)} seqs, overlap TEST {len(set(c)&set(S40))}, overlap DEV {len(set(c)&set(DEV))}")
print("DEV ∩ TEST:", len(set(DEV) & set(S40)))

# ---------- A. JPEG75 under stronger attacks
clean, p10, p50 = L(T+"fp32_clean_nodef"), L(T+"fp32_pgd4s10_nodef"), L(T+"fp32_pgd4s50_nodef")
und_w = np.minimum(v(p10, S40), v(p50, S40))
atk = {"BPDA-10": L(T+"fp32_pgd4s10_jpeg75_adaptive"), "surrogate-10": L(T+"fp32_pgd4s10_jpeg75_adaptive_diff"),
       "surrogate-50": L(T+"fp32_pgd4s50_jpeg75_adaptive_diff")}
jw = np.min(np.stack([v(s, S40) for s in atk.values()]), 0)
c0 = v(clean, S40)
J["A"] = {k: dict(auc=s["success_auc"], gain_same_strength=boot(v(s, S40) - v(p10 if "10" in k else p50, S40))) for k, s in atk.items()}
J["A"]["worst"] = dict(auc=float(jw.mean()), gain=boot(jw - und_w), undefended_worst=float(und_w.mean()),
                       recovered=float((jw.mean() - und_w.mean()) / (c0.mean() - und_w.mean())))
for k, x in J["A"].items(): print("A", k, {kk: (f(vv) if isinstance(vv, dict) else vv) for kk, vv in x.items()})
broken = [i for i, k in enumerate(S40) if c0[i] > .3 and c0[i] - und_w[i] > .2]
J["A"]["broken"] = len(broken)
J["A"]["kept_half_worst_case"] = int(sum(jw[i] >= .5 * c0[i] for i in broken))
J["A"]["kept_half_undefended"] = int(sum(und_w[i] >= .5 * c0[i] for i in broken))
print("broken videos", len(broken), "JPEG worst-case keeps half on", J["A"]["kept_half_worst_case"], "undefended", J["A"]["kept_half_undefended"])
J["A"]["pgd50_vs_pgd10"] = boot(v(p50, S40) - v(p10, S40))
J["A"]["surr10_vs_bpda10"] = boot(v(atk["surrogate-10"], S40) - v(atk["BPDA-10"], S40))
J["A"]["surr50_vs_surr10"] = boot(v(atk["surrogate-50"], S40) - v(atk["surrogate-10"], S40))
print("PGD50 vs PGD10", f(J["A"]["pgd50_vs_pgd10"]), "| surr10 vs BPDA10", f(J["A"]["surr10_vs_bpda10"]), "| surr50 vs surr10", f(J["A"]["surr50_vs_surr10"]))

# ---------- B. DEV sweep + clean cost
dc, dp = L(D+"fp32_clean_nodef"), L(D+"fp32_pgd4s10_nodef")
J["B"] = {}
for q in (75, 85, 90, 95):
    c, a = L(D+f"fp32_clean_jpeg{q}"), L(D+f"fp32_pgd4s10_jpeg{q}_adaptive_diff")
    J["B"][q] = dict(clean=c["success_auc"], clean_change=boot(v(c, DEV) - v(dc, DEV)), robust=a["success_auc"],
                     gain=boot(v(a, DEV) - v(dp, DEV)))
    print(f"DEV q{q}: clean change {f(J['B'][q]['clean_change'])} | robust {a['success_auc']:.4f} gain {f(J['B'][q]['gain'])}")
print(f"DEV clean {dc['success_auc']:.4f}, DEV pgd10 {dp['success_auc']:.4f}")
jc = L(T+"fp32_clean_jpeg75")
tc = v(jc, S40) - c0
J["B"]["test_clean_change"] = boot(tc)
pooled = np.concatenate([tc, v(L(D+"fp32_clean_jpeg75"), DEV) - v(dc, DEV)])
J["B"]["pooled60_clean_change_descriptive"] = boot(pooled)
J["B"]["test_clean_change_without_2_videos_descriptive"] = boot(np.sort(tc)[2:])
print("TEST clean change jpeg75", f(J["B"]["test_clean_change"]), "| pooled DEV+TEST (descriptive)", f(J["B"]["pooled60_clean_change_descriptive"]))

# ---------- D/E. int8 calibration + combined
cpu = L("uav123_fp32_clean_nodef_t4"); cr = v(cpu, S40)
J["D"] = {}
for n in cal40:
    s = L(n); J["D"][n] = dict(auc=s["success_auc"], change=boot(v(s, S40) - cr))
    print("D", n, f"{s['success_auc']:.4f}", f(J["D"][n]["change"]))
ch = [J["D"][n]["change"]["mean"] for n in list(cal40)[:4]]
J["D"]["mean4"], J["D"]["sd4"] = float(np.mean(ch)), float(np.std(ch, ddof=1))
J["D"]["range4"] = [float(min(ch)), float(max(ch))]
print("int8bb 4 calibration sets: mean", J["D"]["mean4"], "SD", J["D"]["sd4"], "range", J["D"]["range4"])
# int8 123 (round 2) for context
s123 = L("uav123_int8bb_clean_nodef_t4"); J["D"]["round2_123"] = boot(v(s123, sorted(s123["per"])) - v(cpu, sorted(s123["per"])))
# how much do sets disagree per video?
M = np.stack([v(L(n), S40) for n in list(cal40)[:4]])
spread = M.max(0) - M.min(0)
J["D"]["videos_spread_gt_0.05"] = int((spread > .05).sum()); J["D"]["max_spread_video"] = [S40[int(spread.argmax())], float(spread.max())]
print("videos where calibration sets disagree by >0.05:", J["D"]["videos_spread_gt_0.05"], "max", J["D"]["max_spread_video"])
comb = L(T+"int8bb_clean_jpegsel_t4")
J["E"] = dict(auc=comb["success_auc"], change=boot(v(comb, S40) - cr),
              vs_same_int8=boot(v(comb, S40) - v(L(T+"int8bb_clean_nodef_t4"), S40)),
              jpeg_ms=comb["ms_per_frame"]["defense"], int8_track_ms=L(T+"int8bb_clean_nodef_t4")["ms_per_frame"]["total"])
same_cal = comb["variant_info"]["calibration_sequences"] == cal40[T+"int8bb_clean_nodef_t4"]
print("E combined", f(J["E"]["change"]), "| JPEG effect on same int8 model", f(J["E"]["vs_same_int8"]), "same calibration as round-2 int8bb:", same_cal)
J["E"]["same_calibration_as_round2_int8bb"] = same_cal
# benches
J["bench"] = {}
for p in sorted((R/"bench").glob("*/summary.json")):
    b = json.loads(p.read_text()); J["bench"][b["bench"]] = {r["defense"]: dict(ms=r["defense_ms"], model_ms=r["model_ms"]) for r in b["rows"]}
print({k: {d: round(x["ms"], 2) for d, x in b.items() if "jpeg" in d} | {"net": round(next(iter(b.values()))["model_ms"], 2)} for k, b in J["bench"].items()})
json.dump(J, open(OUT/"numbers3.json", "w"), indent=1, default=float)

# ================= figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, GR = "#2a6fb0", "#d0703c", "#3b9a6b", "#8c8c8c"
# fig5: attack ladder
fig, ax = plt.subplots(figsize=(6.6, 3.8))
labs = ["BPDA-10\n(round 2)", "surrogate-10", "surrogate-50", "worst\nper video"]
undef = [p10["success_auc"], p10["success_auc"], p50["success_auc"], float(und_w.mean())]
jp = [atk["BPDA-10"]["success_auc"], atk["surrogate-10"]["success_auc"], atk["surrogate-50"]["success_auc"], float(jw.mean())]
x = np.arange(4); w = .36
ax.bar(x - w/2, undef, w, color=GR, label="no defence"); ax.bar(x + w/2, jp, w, color=C1, label="JPEG-75 (attacker knows it)")
ax.axhline(c0.mean(), color="k", lw=.8, ls=":"); ax.text(3.6, c0.mean() + .01, "clean 0.582", ha="right", fontsize=8)
for i in range(4): ax.text(x[i] + w/2, jp[i] + .01, f"{jp[i]:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("success AUC (TEST, 40 videos)"); ax.set_ylim(0, .65)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.1))
fig.tight_layout(); fig.savefig(OUT/"fig5_jpeg_vs_stronger_attacks.png", dpi=160); plt.close(fig)
# fig6: DEV sweep
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for q in (75, 85, 90, 95):
    b = J["B"][q]; ax.errorbar(b["clean_change"]["mean"], b["robust"], xerr=[[b["clean_change"]["mean"] - b["clean_change"]["lo"]], [b["clean_change"]["hi"] - b["clean_change"]["mean"]]],
                               fmt="o", color=C1 if q == 75 else GR, capsize=3)
    ax.annotate(f"q{q}", (b["clean_change"]["mean"], b["robust"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.axvline(-.010, color=C2, ls="--", lw=1); ax.text(-.0105, .285, "clean limit ", color=C2, fontsize=8, ha="right")
ax.axhline(dp["success_auc"], color="k", ls=":", lw=.8); ax.text(.05, dp["success_auc"] + .005, "no defence under attack", fontsize=8, ha="right")
ax.set_xlabel("clean change vs no defence (DEV, 20 videos)"); ax.set_ylabel("AUC under surrogate-10 attack (DEV)")
ax.set_title("Quality chosen on DEV only: q75", loc="left"); fig.tight_layout(); fig.savefig(OUT/"fig6_dev_quality_sweep.png", dpi=160); plt.close(fig)
# fig7: calibration spread
fig, ax = plt.subplots(figsize=(5.6, 3.6))
names = ["round-2 set", "set 1", "set 2", "set 3"]
for i, n in enumerate(list(cal40)[:4]):
    c = J["D"][n]["change"]; ax.errorbar(i, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt="o", color=C3, capsize=3)
ax.axhline(0, color="k", lw=.6); ax.axhline(-.010, color=C2, ls="--", lw=1); ax.text(3.3, -.009, "limit −0.010", color=C2, fontsize=8, ha="right")
ax.axhline(J["D"]["mean4"], color=C3, ls=":", lw=1); ax.text(1.5, J["D"]["mean4"] - .004, f"mean of 4 sets {J['D']['mean4']:+.3f}", color=C3, fontsize=8, ha="center")
ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylabel("int8bb change vs fp32 CPU (TEST)")
ax.set_title("int8 accuracy depends on calibration videos", loc="left"); fig.tight_layout(); fig.savefig(OUT/"fig7_int8_calibration.png", dpi=160); plt.close(fig)
print("figures written")
