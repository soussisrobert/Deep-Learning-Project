"""Round 5 gatekeeper and scorer: failure/attack awareness from HiFT's own confidence, and natural corruptions.

  python scripts/r5_gate.py      # writes results/round5/criteria.md (+ .json); may write plan/round5/*.json

All rules and thresholds are fixed in ROUND5_PLAN.md before any round-5 result exists.
Frame-level quantities are pooled within a video; uncertainty is a bootstrap over VIDEOS (never frames).
Needs the UAV123 ground truth (data/uav123_flat) to compute per-frame overlap, so run it on TC2 via sbatch.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/uav123_flat"
OUT = RES / "round5"
QDIR = ROOT / "plan/round5/queue"
THR_F, ATK_F = ROOT / "plan/round5/threshold.json", ROOT / "plan/round5/attacker_selected.json"
T, D = "uav123-sub40_", "uav123-dev20_"
S40 = sorted(json.loads((ROOT / "attack-subset-40.json").read_text())["sequences"])
DEV = sorted(json.loads((ROOT / "plan/round3/dev-20.json").read_text())["sequences"])
FAIL_IOU = 0.1        # a frame is a tracking failure if overlap with the truth is below 0.1
FALSE_ALARM = 0.05    # alarm threshold: 5th percentile of HiFT's confidence on clean DEV frames
F0_MIN, F_MIN = 0.70, 0.75          # failure-prediction AUROC: DEV pilot / TEST claim
A0_MIN, A_STD_MIN, A_ADAPT_MIN = 0.80, 0.80, 0.50   # share of attack-broken frames that raise the alarm
TEST_FA_MAX = 0.10    # alarm rate on clean TEST frames
EFFECTIVE_DROP = 0.20  # an adaptive attacker "still works" if DEV success AUC drops by >= 0.20
LAMBDAS = (1, 3, 10)
CORR = ["gauss_noise3", "gauss_blur3", "motion_blur3", "contrast3", "low_light3", "low_light5", "night3"]
INT8_PARITY = -0.010

_GT = {}


def gt(names):
    missing = [n for n in names if n not in _GT]
    if missing:
        from hiftstress.data import load_dataset
        for s in load_dataset("flat", str(ROOT / DATA) if not Path(DATA).is_absolute() else DATA, None, missing, None, 0):
            _GT[s.name] = np.asarray(s.gt, float)
    return {n: _GT[n] for n in names}


def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-12)


def frames(run, vids):
    """{video: (iou, best_score, peak, psr)} over tracked frames (1..N-1) where the target is present."""
    d = RES / run
    if not (d / "summary.json").exists():
        return None
    out, G = {}, gt(vids)
    for v in vids:
        p = np.loadtxt(d / "preds" / f"{v}.txt", delimiter=",", ndmin=2)
        c = np.loadtxt(d / "scores" / f"{v}.txt", delimiter=",", ndmin=2)
        g = G[v]
        ok = np.all(np.isfinite(g), 1) & (g[:, 2] > 0) & (g[:, 3] > 0)
        ok[0] = False
        out[v] = (iou(p[ok], g[ok]), c[ok, 0], c[ok, 1], c[ok, 2])
    return out


def auroc(score_fail, score_ok):
    """P(confidence of a failed frame < confidence of a good frame): 1 = perfect warning, 0.5 = useless."""
    x = np.concatenate([score_fail, score_ok])
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r)
    r = (sums / cnt)[inv] + 1
    n1, n0 = len(score_fail), len(score_ok)
    if n1 == 0 or n0 == 0:
        return float("nan")
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(1 - u / (n1 * n0))


def pooled_auroc(fr, vids, col=1):
    f = np.concatenate([fr[v][col][fr[v][0] < FAIL_IOU] for v in vids])
    o = np.concatenate([fr[v][col][fr[v][0] >= FAIL_IOU] for v in vids])
    return auroc(f, o), len(f), len(o)


def boot_videos(fn, vids, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
        x = fn(s)
        if np.isfinite(x):
            vals.append(x)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def flagged_share(fr, vids, thr):
    f = np.concatenate([fr[v][1][fr[v][0] < FAIL_IOU] for v in vids])
    return float((f < thr).mean()) if len(f) else float("nan"), len(f)


def alarm_rate(fr, vids, thr):
    s = np.concatenate([fr[v][1] for v in vids])
    return float((s < thr).mean())


def summ(run):
    p = RES / run / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    s["per"] = {x["name"]: x["success_auc"] for x in json.loads((RES / run / "per_seq.json").read_text())}
    return s


def boot_mean(d, n=10000, seed=0):
    d = np.asarray(d, float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def queue_done(q):
    for line in q.read_text().splitlines():
        name = line.split("\t")[0]
        if not ((RES / name / "summary.json").exists() or (RES / name.split("__p")[0] / "summary.json").exists()):
            return False
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, J, st = ["# Round 5: gated results (ROUND5_PLAN.md)", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", ""], {}, {}
    f3 = lambda x: f"{x:.3f}"  # noqa: E731

    # ===================== P0: DEV pilot (failure warning + alarm against the standard attack)
    md += ["## F/A. Failure and attack awareness from HiFT's own confidence", "",
           f"Failure = frame overlap < {FAIL_IOU}. Alarm = confidence below the 5th percentile of clean DEV frames.", ""]
    dc, dp = frames(D + "fp32_clean_nodef_sc", DEV), frames(D + "fp32_pgd4s10_nodef_sc", DEV)
    if not (dc and dp):
        st["P0"] = "run"
        md.append("- P0 (DEV pilot): not complete")
    else:
        if not THR_F.exists():
            allc = np.concatenate([dc[v][1] for v in DEV])
            THR_F.write_text(json.dumps(dict(threshold=float(np.percentile(allc, 100 * FALSE_ALARM)), source="DEV clean frames",
                                             frames=int(len(allc)), at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
        thr = json.loads(THR_F.read_text())["threshold"]
        a0, nf, no = pooled_auroc(dc, DEV)
        st["F0"] = "pass" if a0 >= F0_MIN else "fail"
        fs0, nfail = flagged_share(dp, DEV, thr)
        st["A0"] = "pass" if fs0 >= A0_MIN else "fail"
        md += [f"- Alarm threshold (fixed from DEV clean): confidence < {thr:.4f}",
               f"- **F0** DEV clean failure-warning AUROC {a0:.3f} ({nf} failed / {no} good frames) → **{st['F0'].upper()}** (≥ {F0_MIN})",
               f"- **A0** DEV standard PGD-10: {fs0:.1%} of {nfail} attack-broken frames raise the alarm → **{st['A0'].upper()}** (≥ {A0_MIN:.0%})"]
        J["P0"] = dict(threshold=thr, auroc=a0, flagged=fs0)
        for col, nm in ((2, "map peak"), (3, "PSR")):
            md.append(f"  - secondary signal {nm}: DEV clean AUROC {pooled_auroc(dc, DEV, col)[0]:.3f}")

    # ===================== P1: choose the strongest adaptive attacker on DEV
    if st.get("A0") == "pass":
        runs = {l: frames(D + f"fp32_pgd4s10shift{l}_nodef_sc", DEV) for l in LAMBDAS}
        if not all(runs.values()):
            st["P1"] = "run"
        else:
            thr = json.loads(THR_F.read_text())["threshold"]
            cl = summ(D + "fp32_clean_nodef_sc")["success_auc"]
            rows = {}
            for l, fr in runs.items():
                a = summ(D + f"fp32_pgd4s10shift{l}_nodef_sc")["success_auc"]
                fs, n = flagged_share(fr, DEV, thr)
                rows[l] = dict(auc=a, drop=cl - a, flagged=fs, failed_frames=n, effective=bool(cl - a >= EFFECTIVE_DROP))
            if not ATK_F.exists():
                eff = [l for l in LAMBDAS if rows[l]["effective"]]
                lam = min(eff, key=lambda l: rows[l]["flagged"]) if eff else min(LAMBDAS, key=lambda l: rows[l]["auc"])
                ATK_F.write_text(json.dumps(dict(shift_lambda=lam, table=rows, rule="strongest evasion among effective attackers (DEV)",
                                                 at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
            sel = json.loads(ATK_F.read_text())
            st["P1"] = "pass"
            md += ["", "| Adaptive attacker λ (DEV) | Success AUC | Drop vs clean | Alarm on broken frames | Effective |", "|---|---|---|---|---|"]
            md += [f"| {l} | {r['auc']:.3f} | {r['drop']:+.3f} | {r['flagged']:.1%} | {'yes' if r['effective'] else 'no'} |" for l, r in rows.items()]
            md.append(f"\nChosen on DEV: λ = **{sel['shift_lambda']}**")
    elif st.get("A0") == "fail":
        md.append("\n**Attack-alarm branch CUT at A0** (the alarm misses the standard attack on DEV).")

    # ===================== TEST
    tc = frames(T + "fp32_clean_nodef_sc", S40)
    if st.get("F0") == "pass" or st.get("A0") == "pass":
        if not tc:
            st["F1"] = "run"
        else:
            a, nf, no = pooled_auroc(tc, S40)
            lo, hi = boot_videos(lambda s: pooled_auroc(tc, s)[0], S40)
            st["F1"] = "pass" if (a >= F_MIN and lo > 0.5) else "fail"
            thr = json.loads(THR_F.read_text())["threshold"]
            far = alarm_rate(tc, S40, thr)
            md += ["", f"- **Claim F** TEST clean failure-warning AUROC **{a:.3f}** [{lo:.3f}, {hi:.3f}] ({nf} failed / {no} good frames) "
                   f"→ **{st['F1'].upper()}** (≥ {F_MIN}, interval above 0.5)",
                   f"- TEST clean false-alarm rate at the DEV threshold: {far:.1%} (≤ {TEST_FA_MAX:.0%} required for Claim A)"]
            J["F1"] = dict(auroc=a, ci=[lo, hi], false_alarm=far)
            if st.get("F0") == "fail":
                st["F1"] = "fail"   # pilot failed: no failure-warning claim, TEST clean only serves the alarm branch
    elif st.get("F0") == "fail" and st.get("A0") == "fail":
        md.append("\n**Both branches CUT on DEV.**")
    if st.get("A0") == "pass":
        ta = frames(T + "fp32_pgd4s10_nodef_sc", S40)
        tadp = frames(T + "fp32_pgd4s10shiftsel_nodef_sc", S40) if ATK_F.exists() else None
        if not ta:
            st["A1"] = "run"
        if ATK_F.exists() and not tadp:
            st["A2"] = "run"
        if ta and tadp and tc:
            thr = json.loads(THR_F.read_text())["threshold"]
            fs1, n1 = flagged_share(ta, S40, thr)
            fs2, n2 = flagged_share(tadp, S40, thr)
            ci1 = boot_videos(lambda s: flagged_share(ta, s, thr)[0], S40)
            ci2 = boot_videos(lambda s: flagged_share(tadp, s, thr)[0], S40)
            far = alarm_rate(tc, S40, thr)
            ok = fs1 >= A_STD_MIN and fs2 >= A_ADAPT_MIN and far <= TEST_FA_MAX
            st["A1"] = st["A2"] = "pass" if ok else "fail"
            sa, sd = summ(T + "fp32_pgd4s10_nodef_sc")["success_auc"], summ(T + "fp32_pgd4s10shiftsel_nodef_sc")["success_auc"]
            md += [f"- **Claim A** alarm on attack-broken TEST frames: standard PGD-10 **{fs1:.1%}** [{ci1[0]:.1%}, {ci1[1]:.1%}] "
                   f"of {n1} (≥ {A_STD_MIN:.0%}); adaptive λ={json.loads(ATK_F.read_text())['shift_lambda']} **{fs2:.1%}** "
                   f"[{ci2[0]:.1%}, {ci2[1]:.1%}] of {n2} (≥ {A_ADAPT_MIN:.0%}); clean false alarms {far:.1%} → **{'PASS' if ok else 'FAIL'}**",
                   f"- Tracking success under attack: standard {sa:.3f}, adaptive {sd:.3f} (clean {summ(T + 'fp32_clean_nodef_sc')['success_auc']:.3f})"]
            J["A"] = dict(standard=fs1, standard_ci=ci1, adaptive=fs2, adaptive_ci=ci2, false_alarm=far, auc_standard=sa, auc_adaptive=sd)

    # ===================== K: corruptions
    md += ["", "## K. Natural corruptions (severity 3 unless stated; synthetic night = darker + sensor noise)", "",
           "| Corruption | fp32 GPU (Δ vs clean) | + JPEG-75 (Δ vs no defence) | int8bb-s1 CPU (Δ vs its clean) | int8 drop − fp32 drop | Failure-warning AUROC (fp32) |",
           "|---|---|---|---|---|---|"]
    clean, i8 = summ(T + "fp32_clean_nodef"), summ(T + "int8sel_clean_nodef_t4_cal1")
    gaps, done = [], 0
    for c in CORR:
        n0, nj, ni = summ(T + f"fp32_{c}_nodef_sc"), summ(T + f"fp32_{c}_jpeg75_sc"), summ(T + f"int8s1_{c}_t4_sc")
        cell = lambda s, ref: (f"{s['success_auc']:.3f} ({boot_mean([s['per'][k] - ref['per'][k] for k in S40])[0]:+.3f} "  # noqa: E731
                               f"[{boot_mean([s['per'][k] - ref['per'][k] for k in S40])[1]:+.3f}, {boot_mean([s['per'][k] - ref['per'][k] for k in S40])[2]:+.3f}])") if (s and ref) else "not run yet"
        gap = None
        if n0 and ni and clean and i8:
            gap = boot_mean([(ni["per"][k] - i8["per"][k]) - (n0["per"][k] - clean["per"][k]) for k in S40])
            gaps.append(gap[0])
            done += 1
        fa = "–"
        fr = frames(T + f"fp32_{c}_nodef_sc", S40) if n0 else None
        if fr:
            fa = f"{pooled_auroc(fr, S40)[0]:.3f}"
        md.append(f"| {c} | {cell(n0, clean)} | {cell(nj, n0)} | {cell(ni, i8)} | "
                  f"{'%+.3f [%+.3f, %+.3f]' % gap if gap else 'not run yet'} | {fa} |")
    if done == len(CORR):
        m = float(np.mean(gaps))
        st["K"] = "pass" if m >= INT8_PARITY else "fail"
        md.append(f"\n**Claim K** (int8 holds up as well as fp32 under bad conditions): mean extra drop {m:+.4f} over "
                  f"{len(CORR)} corruptions → **{st['K'].upper()}** (≥ {INT8_PARITY})")
        J["K"] = dict(mean_extra_drop=m, per=gaps)
    else:
        st["K"] = "run"

    # ===================== R4C: round-4 confirmation (full benchmark + real attribute labels)
    md += ["", "## R4C. Round-4 confirmation: int8bb-s1 on all 123 videos, and losses by official UAV123 attribute", ""]
    NT83 = sorted(json.loads((ROOT / "plan/round5/nontest-83.json").read_text())["sequences"])
    cpu = summ("uav123_fp32_clean_nodef_t4")
    tdraws = [summ(T + f"int8sel_clean_nodef_t4_cal{k}") for k in (1, 2, 3)]
    nt0 = summ("uav123-nt83_int8sel_clean_nodef_t4_cal0")
    nts = [summ(f"uav123-nt83_int8sel30_clean_nodef_t4_cal{k}") for k in (1, 2)]
    change = {}
    if cpu and all(tdraws):
        change.update({k: float(np.mean([d["per"][k] for d in tdraws]) - cpu["per"][k]) for k in S40})
    if cpu and nt0:
        m83 = boot_mean([nt0["per"][k] - cpu["per"][k] for k in NT83])
        md.append(f"- 83 non-TEST videos (calibrated on the 40 TEST videos, exact recipe): {nt0['success_auc']:.4f} vs fp32 CPU "
                  f"{np.mean([cpu['per'][k] for k in NT83]):.4f}: change {m83[0]:+.4f} [{m83[1]:+.4f}, {m83[2]:+.4f}]")
        for k2, r in zip((1, 2), nts):
            if r:
                x = boot_mean([r["per"][k] - cpu["per"][k] for k in NT83])
                md.append(f"  - sensitivity draw {k2} (30 calibration videos): change {x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]")
        change.update({k: nt0["per"][k] - cpu["per"][k] for k in NT83})
    if len(change) == 123:
        allm = boot_mean(list(change.values()))
        st["E123"] = "pass" if allm[0] >= -0.010 else "fail"
        worst = sorted(change.items(), key=lambda kv: kv[1])[:5]
        md += [f"- **Claim E-full** all 123 videos (TEST: mean of 3 draws; others: exact recipe): change **{allm[0]:+.4f}** "
               f"[{allm[1]:+.4f}, {allm[2]:+.4f}] → **{st['E123'].upper()}** (≥ −0.010); "
               f"int8 {np.mean([cpu['per'][k] + change[k] for k in change]):.4f} vs fp32 CPU {cpu['success_auc']:.4f}",
               "- Largest losses: " + ", ".join(f"{k} {v:+.3f}" for k, v in worst)]
        J["E123"] = dict(change=allm, worst=worst)
    else:
        st["E123"] = "run"
    af = RES / "attributes_per_video.json"
    if af.exists() and len(change) == 123:
        A = json.loads(af.read_text())["videos"]
        names = sorted({a for v in A.values() for a, f in v.items() if f})
        md += ["", f"Official attribute labels ({json.loads(af.read_text())['source']}). Descriptive, not a claim:", "",
               "| Attribute | Videos | int8 change: with | without | difference [95%] |", "|---|---|---|---|---|"]
        rng = np.random.default_rng(0)
        for a in names:
            w = [change[k] for k in change if A.get(k, {}).get(a)]
            wo = [change[k] for k in change if not A.get(k, {}).get(a)]
            if len(w) < 5 or len(wo) < 5:
                continue
            bs = [np.mean(rng.choice(w, len(w))) - np.mean(rng.choice(wo, len(wo))) for _ in range(5000)]
            md.append(f"| {a} | {len(w)} | {np.mean(w):+.4f} | {np.mean(wo):+.4f} | {np.mean(w) - np.mean(wo):+.4f} "
                      f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] |")
    elif not af.exists():
        st["R4C1"] = "run"

    # ===================== claims + NEXT
    md += ["", "## Claims", ""]
    for k, keys in (("Claim F (HiFT's confidence warns of tracking failure)", ("F0", "F1")),
                    ("Claim A (low-confidence alarm catches attacks, including an attacker who knows about it)", ("A0", "A1")),
                    ("Claim K (int8 keeps up with fp32 under natural corruptions)", ("K",)),
                    ("Claim E-full (round-4 int8 recipe within −0.010 on all 123 videos)", ("E123",))):
        v = "not met" if any(st.get(x) == "fail" for x in keys) else ("pending" if any(st.get(x) in (None, "run") for x in keys) else "SUPPORTED")
        md.append(f"- {k}: **{v}**")
    want = {"R4C1": True, "R4C2": True, "P0": st.get("P0") == "run", "K1": True, "K2": True,
            "P1": st.get("P1") == "run", "F1": st.get("F1") == "run",
            "A1": st.get("A1") == "run", "A2": st.get("A2") == "run" and ATK_F.exists()}
    nxt = []
    for step, on in want.items():
        for q in sorted(QDIR.glob("*.tsv")):
            if (q.name.split("_")[0] == step or q.name.startswith(step + "-")) and on and not queue_done(q):
                nxt.append(str(q.relative_to(ROOT)))
    md += ["", "## NEXT (one GPU job at a time; a CPU queue may run beside it)", ""]
    md += [f"- `{q}`" for q in nxt] or ["- nothing: round 5 is finished. Package the results."]
    J["status"], J["next"] = st, nxt
    (OUT / "criteria.md").write_text("\n".join(md) + "\n")
    (OUT / "criteria.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
