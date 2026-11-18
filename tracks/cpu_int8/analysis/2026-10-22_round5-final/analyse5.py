"""Round-5 independent analysis (saved results + supplied ground truth only)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PK, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
R, SRC, GTD = PK/"results", PK/"source", PK/"ground-truth"
S40 = sorted(json.loads((SRC/"attack-subset-40.json").read_text())["sequences"])
DEV = sorted(json.loads((SRC/"plan/round3/dev-20.json").read_text())["sequences"])
NT83 = sorted(json.loads((SRC/"plan/round5/nontest-83.json").read_text())["sequences"])
T, D = "uav123-sub40_", "uav123-dev20_"
THR = json.loads((SRC/"plan/round5/threshold.json").read_text())["threshold"]
J, problems = {}, []
def per(n): return {p["name"]: p["success_auc"] for p in json.loads((R/n/"per_seq.json").read_text())}
def boot(d, n=10000, seed=0):
    d = np.asarray(d, float); bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                median=float(np.median(d)), better=int((d > .05).sum()), worse=int((d < -.05).sum()))
f = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (median {p['median']:+.4f}; {p['better']} up / {p['worse']} down by >0.05)"
def gt(v):
    g = np.loadtxt(GTD/f"{v}.txt", delimiter=",", ndmin=2) if (GTD/f"{v}.txt").read_text().count(",") else np.loadtxt(GTD/f"{v}.txt", ndmin=2)
    return g
def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 0]+a[:, 2], b[:, 0]+b[:, 2]), np.minimum(a[:, 1]+a[:, 3], b[:, 1]+b[:, 3])
    inter = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    return inter/(a[:, 2]*a[:, 3]+b[:, 2]*b[:, 3]-inter+1e-12)
def frames(run, vids):
    out = {}
    for v in vids:
        p = np.loadtxt(R/run/"preds"/f"{v}.txt", delimiter=",", ndmin=2)
        c = np.genfromtxt(R/run/"scores"/f"{v}.txt", delimiter=",", comments="#")
        if c.ndim == 1: c = c[None]
        g = gt(v)
        if len(p) != len(g) or len(c) != len(g): problems.append(f"{run}/{v}: rows {len(p)} {len(c)} {len(g)}")
        ok = np.all(np.isfinite(g), 1) & (g[:, 2] > 0) & (g[:, 3] > 0); ok[0] = False
        out[v] = dict(iou=iou(p[ok], g[ok]), s=c[ok, 0], idx=np.where(ok)[0])
    return out
def auroc(fail, good):
    x = np.concatenate([fail, good]); r = np.argsort(np.argsort(x, kind="mergesort")).astype(float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True); r = (np.bincount(inv, r)/cnt)[inv]+1
    n1, n0 = len(fail), len(good); return float(1-(r[:n1].sum()-n1*(n1+1)/2)/(n1*n0))
def pooled(fr, vids):
    fa = np.concatenate([fr[v]["s"][fr[v]["iou"] < .1] for v in vids]); go = np.concatenate([fr[v]["s"][fr[v]["iou"] >= .1] for v in vids])
    return auroc(fa, go), len(fa), len(go)
def bootv(fn, vids, n=2000, seed=0):
    rng = np.random.default_rng(seed); vals = [fn([vids[i] for i in rng.integers(0, len(vids), len(vids))]) for _ in range(n)]
    vals = [x for x in vals if np.isfinite(x)]; return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

# ---------- E-full
cpu = per("uav123_fp32_clean_nodef_t4")
td = [per(T+f"int8sel_clean_nodef_t4_cal{k}") for k in (1, 2, 3)]
nt0 = per("uav123-nt83_int8sel_clean_nodef_t4_cal0"); nts = [per(f"uav123-nt83_int8sel30_clean_nodef_t4_cal{k}") for k in (1, 2)]
for n, d, exp in [("nt0", nt0, NT83)] + [(f"nts{k}", x, NT83) for k, x in enumerate(nts, 1)]:
    if sorted(d) != exp: problems.append(f"{n}: video set")
ch = {k: float(np.mean([d[k] for d in td]) - cpu[k]) for k in S40}; ch.update({k: nt0[k]-cpu[k] for k in NT83})
J["E"] = dict(all123=boot(list(ch.values())), test40=boot([ch[k] for k in S40]), nt83=boot([ch[k] for k in NT83]),
              nt83_sens=[boot([x[k]-cpu[k] for k in NT83]) for x in nts])
for k, v in J["E"].items(): print("E", k, f(v) if isinstance(v, dict) else [f(x) for x in v])
srt = sorted(ch.items(), key=lambda kv: kv[1]); J["E"]["worst"] = srt[:6]; J["E"]["best"] = srt[-4:]
print("worst", [(k, round(v, 3)) for k, v in srt[:6]], "best", [(k, round(v, 3)) for k, v in srt[-4:]])
dv = np.array(list(ch.values())); J["E"]["within_002"] = int((abs(dv) <= .02).sum())
print("videos within ±0.02:", J["E"]["within_002"], "/ 123; >0.05 up", int((dv > .05).sum()), "down", int((dv < -.05).sum()))
# does the 83 split mirror the 40?  CPU vs GPU fp32 on 83 (noise floor)
gpu = per("uav123_fp32_clean_nodef")
J["E"]["noise_cpu_vs_gpu_123"] = boot([cpu[k]-gpu[k] for k in cpu])
print("noise floor fp32 CPU vs GPU (123):", f(J["E"]["noise_cpu_vs_gpu_123"]))
x = np.array([v for k, v in ch.items() if k not in ("person19_2", "uav4")]); J["E"]["all121_excl_two_flips"] = boot(x); print("E excl person19_2, uav4:", f(J["E"]["all121_excl_two_flips"]))
# speed recap
e = {p.stem: json.loads(p.read_text()) for p in (R/"edge_round4").glob("*.json")}
J["speed"] = dict(net_t4=e["cpu_t4_fp32"]["latency_ms_mean"]/e["cpu_t4_int8bb-s1"]["latency_ms_mean"], fp32=e["cpu_t4_fp32"]["latency_ms_mean"], s1=e["cpu_t4_int8bb-s1"]["latency_ms_mean"])
nt0s = json.loads((R/"uav123-nt83_int8sel_clean_nodef_t4_cal0"/"summary.json").read_text())["ms_per_frame"]
J["speed"]["nt83_track_ms"] = nt0s["total"]; print("speed", J["speed"])

# ---------- F
dc = frames(D+"fp32_clean_nodef_sc", DEV); tc = frames(T+"fp32_clean_nodef_sc", S40)
a, nf, ng = pooled(tc, S40); ci = bootv(lambda s: pooled(tc, s)[0], S40)
J["F"] = dict(test_auroc=a, ci=ci, failed=nf, good=ng, dev_auroc=pooled(dc, DEV)[0])
# per-video: how concentrated are failures?  share of failed frames from top-5 videos
fv = sorted(((v, int((tc[v]["iou"] < .1).sum())) for v in S40), key=lambda x: -x[1])
J["F"]["failed_frames_top5"] = fv[:5]; J["F"]["videos_with_failures"] = int(sum(1 for _, n in fv if n > 0))
# within-video AUROC (does it work inside a video, not just across easy/hard videos?)
wv = [auroc(tc[v]["s"][tc[v]["iou"] < .1], tc[v]["s"][tc[v]["iou"] >= .1]) for v in S40 if 20 <= (tc[v]["iou"] < .1).sum() <= len(tc[v]["iou"]) - 20]
J["F"]["within_video_auroc_median"] = float(np.median(wv)) if wv else None; J["F"]["within_video_n"] = len(wv)
# at the fixed alarm threshold
allf = np.concatenate([tc[v]["s"][tc[v]["iou"] < .1] for v in S40]); allg = np.concatenate([tc[v]["s"][tc[v]["iou"] >= .1] for v in S40])
J["F"]["recall_at_thr"] = float((allf < THR).mean()); J["F"]["false_alarm_at_thr"] = float((allg < THR).mean())
J["F"]["precision_at_thr"] = float((allf < THR).sum()/max(1, (allf < THR).sum()+(allg < THR).sum()))
# lead time: for each failure onset (first failed frame after >=10 good frames), was the alarm already on in the 5 frames before?
lead, onsets = 0, 0
for v in S40:
    io, s = tc[v]["iou"], tc[v]["s"]
    for i in range(10, len(io)):
        if io[i] < .1 and np.all(io[i-10:i] >= .1):
            onsets += 1; lead += int(np.any(s[i-5:i] < THR))
J["F"]["onsets"], J["F"]["alarm_before_onset"] = onsets, lead
print("F:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in J["F"].items()})
# int8 confidence under corruption (descriptive): scores saved in K2
J["F"]["int8_under_corruption"] = {}
for c in ("gauss_noise3", "gauss_blur3", "motion_blur3", "contrast3", "low_light3", "low_light5", "night3"):
    fr8 = frames(T+f"int8s1_{c}_t4_sc", S40); fr32 = frames(T+f"fp32_{c}_nodef_sc", S40)
    J["F"]["int8_under_corruption"][c] = dict(int8=pooled(fr8, S40)[0], fp32=pooled(fr32, S40)[0])
print("AUROC under corruption (fp32 / int8):", {c: (round(x["fp32"], 3), round(x["int8"], 3)) for c, x in J["F"]["int8_under_corruption"].items()})

# ---------- A: why the alarm failed (DEV only; no TEST attack runs exist)
dp = frames(D+"fp32_pgd4s10_nodef_sc", DEV)
broken = np.concatenate([dp[v]["s"][dp[v]["iou"] < .1] for v in DEV]); cleang = np.concatenate([dc[v]["s"][dc[v]["iou"] >= .1] for v in DEV])
cleanf = np.concatenate([dc[v]["s"][dc[v]["iou"] < .1] for v in DEV])
J["A"] = dict(broken_frames=len(broken), flagged=float((broken < THR).mean()), median_conf_broken=float(np.median(broken)),
              median_conf_clean_good=float(np.median(cleang)), median_conf_clean_failed=float(np.median(cleanf)),
              share_broken_above_clean_good_median=float((broken > np.median(cleang)).mean()),
              exploratory_auroc_high_conf_flags_attack=1-auroc(broken, cleang))
print("A:", {k: round(v, 3) if isinstance(v, float) else v for k, v in J["A"].items()})

# ---------- K
clean, i8 = per(T+"fp32_clean_nodef"), per(T+"int8sel_clean_nodef_t4_cal1")
J["K"] = {}
for c in ("gauss_noise3", "gauss_blur3", "motion_blur3", "contrast3", "low_light3", "low_light5", "night3"):
    n0, nj, ni = per(T+f"fp32_{c}_nodef_sc"), per(T+f"fp32_{c}_jpeg75_sc"), per(T+f"int8s1_{c}_t4_sc")
    J["K"][c] = dict(fp32_drop=boot([n0[k]-clean[k] for k in S40]), int8_drop=boot([ni[k]-i8[k] for k in S40]),
                     extra=boot([(ni[k]-i8[k])-(n0[k]-clean[k]) for k in S40]), jpeg=boot([nj[k]-n0[k] for k in S40]))
for c, x in J["K"].items(): print(f"K {c}: fp32 {x['fp32_drop']['mean']:+.3f} int8 {x['int8_drop']['mean']:+.3f} extra {f(x['extra'])} jpeg {x['jpeg']['mean']:+.3f}")
ex = [x["extra"]["mean"] for x in J["K"].values()]; J["K_mean_extra"] = float(np.mean(ex)); print("K mean extra", np.mean(ex))
J["problems"] = problems; print("problems:", problems or "none")
json.dump(J, open(OUT/"numbers5.json", "w"), indent=1, default=float)

# ---------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, GR = "#2a6fb0", "#d0703c", "#3b9a6b", "#8c8c8c"
fig, ax = plt.subplots(figsize=(6.6, 3.8))
bins = np.linspace(min(cleang.min(), broken.min(), cleanf.min()), max(cleang.max(), broken.max()), 60)
ax.hist(cleang, bins, density=True, alpha=.55, color=C3, label="clean, tracking OK")
ax.hist(cleanf, bins, density=True, alpha=.55, color=C2, label="clean, tracking failed")
ax.hist(broken, bins, density=True, alpha=.55, color=C1, label="attacked, tracking failed")
ax.axvline(THR, color="k", ls="--", lw=1); ax.text(THR, ax.get_ylim()[1]*.9, " alarm threshold", fontsize=8)
ax.set_xlabel("HiFT confidence (best_score), DEV frames"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Natural failures often look less sure; attacked failures look as sure as good tracking", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"fig11_confidence.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
names = list(J["K"]); x = np.arange(len(names)); w = .38
ax.bar(x-w/2, [J["K"][c]["fp32_drop"]["mean"] for c in names], w, color=GR, label="fp32 (GPU) drop")
ax.bar(x+w/2, [J["K"][c]["int8_drop"]["mean"] for c in names], w, color=C1, label="int8bb-s1 (CPU) drop")
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=8, rotation=20)
ax.set_ylabel("change in success AUC vs own clean (TEST)"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
ax.set_title("Corruptions hurt both models similarly (int8 a bit worse on noise, very dark)", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"fig12_corruptions.png", dpi=160); plt.close(fig)
print("figures written")
