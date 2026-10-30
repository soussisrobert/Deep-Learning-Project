"""Round-2 final analysis: independent paired statistics and figures (reads saved results only)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S40 = json.loads((R.parent / "source/attack-subset-40.json").read_text())["sequences"]
P = "uav123-sub40_"
def L(n):
    s = json.loads((R/n/"summary.json").read_text())
    s["per"] = {p["name"]: p for p in json.loads((R/n/"per_seq.json").read_text())}
    return s
def auc(s, seqs): return float(np.mean([s["per"][k]["success_auc"] for k in seqs]))
def paired(a, b, seqs=None, n=10000, seed=0):
    seqs = sorted(seqs or (set(a["per"]) & set(b["per"])))
    d = np.array([a["per"][k]["success_auc"] - b["per"][k]["success_auc"] for k in seqs])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), better=int((d > .05).sum()), worse=int((d < -.05).sum()),
                p_le=float((bs <= 0).mean()), d={k: float(v) for k, v in zip(seqs, d)})
f = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (n={p['n']}, +{p['better']}/-{p['worse']})"
J = {}
clean, pgd = L(P+"fp32_clean_nodef"), L(P+"fp32_pgd4s10_nodef")
lost = clean["success_auc"] - pgd["success_auc"]
print(f"clean40 {clean['success_auc']:.4f} pgd40 {pgd['success_auc']:.4f} lost {lost:.4f}")

# ---------- defences
D = {}
for d in ["blur1", "median3", "bits5", "jpeg75"]:
    c, u, a = L(P+f"fp32_clean_{d}"), L(P+f"fp32_pgd4s10_{d}"), L(P+f"fp32_pgd4s10_{d}_adaptive")
    pc, pu, pa = paired(c, clean), paired(u, pgd), paired(a, pgd)
    adapt_vs_unaware = paired(a, u)
    D[d] = dict(clean=c["success_auc"], unaware=u["success_auc"], adaptive=a["success_auc"], clean_cost=pc, gain_unaware=pu,
                gain_adaptive=pa, adaptive_minus_unaware=adapt_vs_unaware, recovered=pa["mean"] / lost,
                defense_ms=c["ms_per_frame"]["defense"], overhead=c["ms_per_frame"]["defense"] / clean["ms_per_frame"]["total"],
                net_adaptive_vs_clean=paired(a, clean))
    print(f"{d:8s} clean {f(pc)} | unaware {f(pu)} | adaptive {f(pa)} | recov {pa['mean']/lost:.0%} | adapt-unaware {adapt_vs_unaware['mean']:+.3f} | def {D[d]['defense_ms']:.2f} ms ({D[d]['overhead']:.1%})")
J["defences"] = D
# JPEG: which videos pay the clean cost
jc = D["jpeg75"]["clean_cost"]["d"]
print("jpeg75 clean cost, worst 5:", sorted(jc.items(), key=lambda kv: kv[1])[:5], " best 3:", sorted(jc.items(), key=lambda kv: -kv[1])[:3])
print("jpeg75 clean: share of videos within ±0.01:", np.mean([abs(v) <= .01 for v in jc.values()]))
ja = D["jpeg75"]["gain_adaptive"]["d"]
print("jpeg75 adaptive gain: videos with gain >0.05:", sum(v > .05 for v in ja.values()), " median gain", np.median(list(ja.values())))
# Videos that PGD actually breaks (clean AUC>0.3 and pgd loss >0.2): how many does jpeg restore to >= half clean?
broken = [k for k in S40 if clean["per"][k]["success_auc"] > .3 and clean["per"][k]["success_auc"] - pgd["per"][k]["success_auc"] > .2]
ja_s = L(P+"fp32_pgd4s10_jpeg75_adaptive")
rest = [k for k in broken if ja_s["per"][k]["success_auc"] >= .5 * clean["per"][k]["success_auc"]]
J["jpeg_restored"] = dict(broken=len(broken), restored_half=len(rest))
print(f"videos broken by PGD: {len(broken)}; jpeg75 adaptive keeps >= half of clean AUC on {len(rest)}")
for d in ["blur1", "median3", "bits5"]:
    s = L(P+f"fp32_pgd4s10_{d}_adaptive")
    print(f"  {d}: keeps half on {sum(s['per'][k]['success_auc'] >= .5*clean['per'][k]['success_auc'] for k in broken)}")
print(f"  none: keeps half on {sum(pgd['per'][k]['success_auc'] >= .5*clean['per'][k]['success_auc'] for k in broken)}")

# ---------- compression
cpu123 = L("uav123_fp32_clean_nodef_t4")
C = {}
for v in ["int8bb", "int8full", "prune50"]:
    s40 = L(P+f"{v}_clean_nodef_t4")
    C[v] = dict(acc40=paired(s40, cpu123, S40))
    try:
        s123 = L(f"uav123_{v}_clean_nodef_t4")
        cal = set(s123["variant_info"]["calibration_sequences"])
        C[v]["acc123"] = paired(s123, cpu123)
        C[v]["acc123_excl_calib"] = paired(s123, cpu123, set(s123["per"]) - cal)
        C[v]["acc123_on_fixed40"] = paired(s123, cpu123, S40)
        C[v]["acc123_on_other83"] = paired(s123, cpu123, set(s123["per"]) - set(S40))
        C[v]["acc123_on_other83_excl_calib"] = paired(s123, cpu123, set(s123["per"]) - set(S40) - cal)
        C[v]["run40_vs_run123_on_fixed40"] = paired(s40, s123, S40)
        C[v]["p20_official_123"] = s123["precision_20_official"]
    except FileNotFoundError:
        pass
    print(v, {k: f(x) for k, x in C[v].items() if isinstance(x, dict)})
for v in ["int8bb", "int8full"]:
    a = L(P+f"{v}_pgd4s10_nodef_t4")
    C[v]["pgd_vs_fp32gpu_pgd"] = paired(a, pgd)
    C[v]["pgd_loss_vs_own_clean"] = paired(a, L(P+f"{v}_clean_nodef_t4"))
    print(v, "pgd", f(C[v]["pgd_vs_fp32gpu_pgd"]), "own loss", f(C[v]["pgd_loss_vs_own_clean"]))
E = {}
for sub in ("edge", "edge_round2"):
    for p in sorted((R/sub).glob("cpu_t*_*.json")):
        j = json.loads(p.read_text()); E[(sub, j["threads"], j["variant"])] = j
for k, j in sorted(E.items()):
    print(k, f"{j['latency_ms_mean']:.1f} ms  size {j['size_mb']:.1f}  peak {j.get('same_peak_fraction')}")
J["compression"] = C
J["edge"] = {f"{a}|t{b}|{c}": dict(ms=j["latency_ms_mean"], size=j["size_mb"], peak=j.get("same_peak_fraction"),
                                     backbone=j.get("backbone_ms_mean")) for (a, b, c), j in E.items()}

# ---------- attacks
A = {}
for kind, eps_list in [("noise", [1, 2, 4, 8]), ("fgsm", [1, 2, 4, 8]), ("pgd", [1, 2, 4, 8])]:
    for e in eps_list:
        n = P + (f"fp32_{kind}{e}s10_nodef" if kind == "pgd" else f"fp32_{kind}{e}_nodef")
        if (R/n).exists():
            A[f"{kind}{e}"] = L(n)["success_auc"]
seeds = {k: [L(P+f"fp32_{k}_nodef{s}")["success_auc"] for s in ("", "_seed1", "_seed2")] for k in ("pgd4s10", "noise4")}
print("attack curve", A, "seeds", {k: (np.mean(v), np.std(v, ddof=1)) for k, v in seeds.items()})
J["attacks"], J["seeds"] = A, seeds
# official precision for clean refs
J["official_p20"] = {n: L(n).get("precision_20_official") for n in ["uav123_int8bb_clean_nodef_t4", "uav123_int8full_clean_nodef_t4"]}
json.dump(J, open(OUT/"numbers.json", "w"), indent=1, default=float)

# ================= figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4, GR = "#2a6fb0", "#d0703c", "#3b9a6b", "#8a5cb8", "#8c8c8c"
# Fig 1: attack curve
fig, ax = plt.subplots(figsize=(6, 3.8))
for kind, col, lab in [("noise", GR, "random noise (control)"), ("fgsm", C2, "FGSM (1 step)"), ("pgd", C1, "PGD (10 steps)")]:
    xs = [e for e in (1, 2, 4, 8) if f"{kind}{e}" in A]
    ax.plot([0] + xs, [clean["success_auc"]] + [A[f"{kind}{e}"] for e in xs], "o-", color=col, label=lab)
ax.errorbar([4], [np.mean(seeds["pgd4s10"])], yerr=[np.std(seeds["pgd4s10"], ddof=1)], fmt="none", ecolor=C1, capsize=4)
ax.set_xticks([0, 1, 2, 4, 8]); ax.set_xlabel("perturbation budget ε (pixel levels out of 255)"); ax.set_ylabel("success AUC (40 videos)")
ax.set_ylim(0, .65); ax.legend(frameon=False); ax.set_title("A targeted attack, not noise, breaks HiFT", loc="left")
fig.tight_layout(); fig.savefig(OUT/"fig1_attack_curve.png", dpi=160); plt.close(fig)
# Fig 2: defence trade-off
fig, ax = plt.subplots(figsize=(7, 3.9))
names = ["none", "blur1", "median3", "bits5", "jpeg75"]
vals = {"clean": [clean["success_auc"]] + [D[d]["clean"] for d in names[1:]],
        "PGD, attacker unaware": [pgd["success_auc"]] + [D[d]["unaware"] for d in names[1:]],
        "PGD, adaptive attacker": [pgd["success_auc"]] + [D[d]["adaptive"] for d in names[1:]]}
w = .26
for i, (lab, col) in enumerate(zip(vals, [GR, "#9cc3e6", C1])):
    ax.bar(np.arange(5) + (i - 1) * w, vals[lab], w, color=col, label=lab)
ax.axhline(clean["success_auc"] - .010, color=C2, lw=1, ls="--"); ax.text(-.45, clean["success_auc"] - .002, "clean limit (−0.010)", color=C2, fontsize=8, va="top")
ax.set_xticks(range(5)); ax.set_xticklabels([f"{n}\n{D[n]['defense_ms']:.2f} ms" if n != "none" else "no defence" for n in names])
ax.set_ylabel("success AUC (40 videos)"); ax.set_ylim(0, .65); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.12))
fig.tight_layout(); fig.savefig(OUT/"fig2_defences.png", dpi=160); plt.close(fig)
# Fig 3: per-video jpeg adaptive vs pgd
fig, ax = plt.subplots(figsize=(4.6, 4.2))
x = [clean["per"][k]["success_auc"] for k in S40]
ax.scatter(x, [pgd["per"][k]["success_auc"] for k in S40], s=18, color=GR, label="PGD, no defence")
ax.scatter(x, [ja_s["per"][k]["success_auc"] for k in S40], s=18, color=C1, label="PGD adaptive + JPEG75")
ax.plot([0, 1], [0, 1], color="k", lw=.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("clean AUC (per video)"); ax.set_ylabel("AUC under attack"); ax.legend(frameon=False, fontsize=8)
ax.set_title("JPEG75 per video (40)", loc="left"); fig.tight_layout(); fig.savefig(OUT/"fig3_jpeg_per_video.png", dpi=160); plt.close(fig)
# Fig 4: compression speed vs accuracy (123 where available)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
f32 = E[("edge_round2", 4, "fp32")]["latency_ms_mean"]
pts = [("fp32", f32, cpu123["success_auc"], GR), ("int8bb", E[("edge_round2", 4, "int8bb")]["latency_ms_mean"], L("uav123_int8bb_clean_nodef_t4")["success_auc"], C3),
       ("int8full", E[("edge_round2", 4, "int8full")]["latency_ms_mean"], L("uav123_int8full_clean_nodef_t4")["success_auc"], C2),
       ("prune50*", E[("edge", 4, "prune50")]["latency_ms_mean"], cpu123["success_auc"] + C["prune50"]["acc40"]["mean"], C4)]
for n, ms, a, col in pts:
    ax.scatter(ms, a, s=60, color=col); ax.annotate(f"{n}\n{f32/ms:.2f}×", (ms, a), textcoords="offset points", xytext=(6, -4), fontsize=8)
ax.axhspan(cpu123["success_auc"] - .010, cpu123["success_auc"] + .010, color=C3, alpha=.08)
ax.set_xlabel("network latency, 4 Xeon threads (ms)"); ax.set_ylabel("success AUC (UAV123, 123 videos)")
ax.set_xlim(30, 100); ax.set_title("int8 backbone: 2.1× faster, same accuracy", loc="left")
fig.text(.01, .01, "*prune50: Δ measured on the 40-video subset, added to the fp32 123-video score", fontsize=7, color=GR)
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT/"fig4_compression.png", dpi=160); plt.close(fig)
print("figures written")
