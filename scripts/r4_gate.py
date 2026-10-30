"""Round 4 gatekeeper and scorer: what passed, what is cut, and which queue to submit next.

  python scripts/r4_gate.py              # writes results/round4/criteria.md (+ .json); may write plan/round4/int8_selected.json

Every rule and threshold is fixed in ROUND4_PLAN.md before any round-4 result exists.
A branch that fails a gate is CUT: its later queues are never listed under NEXT.
Paired by video, 95% bootstrap over videos (10,000 resamples).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = RES / "round4"
QDIR = ROOT / "plan/round4/queue"
SELF = ROOT / "plan/round4/int8_selected.json"
CLEAN_COST_MAX, GAIN_MIN, OVERHEAD_MAX, SPEEDUP_MIN, WORST_DRAW_MIN = 0.010, 0.10, 0.0385, 1.5, -0.020
T, D = "uav123-sub40_", "uav123-dev20_"
S40 = sorted(json.loads((ROOT / "attack-subset-40.json").read_text())["sequences"])
DEV = sorted(json.loads((ROOT / "plan/round3/dev-20.json").read_text())["sequences"])
RECIPES = {"int8bb_c20x2": ("int8bb", "--calib-seqs 20 --calib-frames 2"),
           "int8bb_c40x4s": ("int8bb", "--calib-seqs 40 --calib-frames 4 --calib-spread"),
           "int8bb-s1_c20x2": ("int8bb-s1", "--calib-seqs 20 --calib-frames 2"),
           "int8bb-s1_c40x4s": ("int8bb-s1", "--calib-seqs 40 --calib-frames 4 --calib-spread")}


def L(n):
    d = RES / n
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    s["per"] = {p["name"]: p["success_auc"] for p in json.loads((d / "per_seq.json").read_text())}
    return s


def v(s, keys):
    return np.array([s["per"][k] for k in keys])


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), better=int((d > .05).sum()), worse=int((d < -.05).sum()))


def f(p):
    return f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] (+{p['better']}/−{p['worse']})"


def latency(variant, threads=4):
    p = RES / "edge_round4" / f"cpu_t{threads}_{variant}.json"
    return json.loads(p.read_text())["latency_ms_mean"] if p.exists() else None


def queue_done(q):
    for line in q.read_text().splitlines():
        name = line.split("\t")[0]
        base = name.split("__p")[0]
        if not ((RES / name / "summary.json").exists() or (RES / base / "summary.json").exists()):
            return False
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, J, status = ["# Round 4: gated results (ROUND4_PLAN.md)", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", ""], {}, {}
    clean, p10, p50 = L(T + "fp32_clean_nodef"), L(T + "fp32_pgd4s10_nodef"), L(T + "fp32_pgd4s50_nodef")
    cpu123 = L("uav123_fp32_clean_nodef_t4")
    if not (clean and p10 and p50 and cpu123):
        raise SystemExit("round-2/3 baselines missing (fp32 clean, PGD-10, PGD-50, fp32 CPU 123)")

    # ================= J: JPEG-85
    md += ["## J. JPEG-85 (quality picked from round-3 DEV evidence; first TEST use)", ""]
    j1 = L(T + "fp32_clean_jpeg85")
    if not j1:
        status["J1"] = "run"
    else:
        h7 = boot(v(j1, S40) - v(clean, S40))
        status["J1"] = "pass" if h7["mean"] >= -CLEAN_COST_MAX else "fail"
        md.append(f"- **J1 / H7 clean cost** (GPU, TEST): {j1['success_auc']:.4f}, change {f(h7)} → **{status['J1'].upper()}** (≥ −0.010)")
        J["H7"] = h7
    jc, jn = L(T + "fp32_clean_jpeg85_t4"), L(T + "fp32_clean_nodef_t4")
    if jc and jn:
        sh = jc["ms_per_frame"]["defense"] / jn["ms_per_frame"]["total"]
        status["H8cpu"] = "pass" if sh <= OVERHEAD_MAX else "fail"
        md.append(f"- **H8 overhead, CPU** (fp32, 4 threads): JPEG {jc['ms_per_frame']['defense']:.2f} ms / tracking "
                  f"{jn['ms_per_frame']['total']:.2f} ms = {sh:.1%} → **{status['H8cpu'].upper()}** (≤ 3.85%). "
                  f"Clean on CPU: change {f(boot(v(jc, S40) - v(jn, S40)))}")
        J["H8cpu"] = sh
    if j1:
        shg = j1["ms_per_frame"]["defense"] / clean["ms_per_frame"]["total"]
        md.append(f"- Overhead on the A40 GPU (reported, not a round-4 claim): {j1['ms_per_frame']['defense']:.2f} ms = {shg:.1%}")
    j2 = L(T + "fp32_pgd4s10_jpeg85_adaptive_diff")
    if status.get("J1") == "pass":
        if not j2:
            status["J2"] = "run"
        else:
            g = boot(v(j2, S40) - v(p10, S40))
            status["J2"] = "pass" if g["mean"] >= GAIN_MIN and g["lo"] > 0 else "fail"
            md.append(f"- **J2 surrogate-10 gate**: {j2['success_auc']:.4f}, gain vs undefended PGD-10 {f(g)} → **{status['J2'].upper()}**")
    if status.get("J2") == "pass":
        s50, b10 = L(T + "fp32_pgd4s50_jpeg85_adaptive_diff"), L(T + "fp32_pgd4s10_jpeg85_adaptive")
        if not (s50 and b10):
            status["J3"] = "run"
        else:
            w = np.min(np.stack([v(x, S40) for x in (j2, s50, b10)]), 0)
            uw = np.minimum(v(p10, S40), v(p50, S40))
            h6 = boot(w - uw)
            status["J3"] = "pass" if h6["mean"] >= GAIN_MIN and h6["lo"] > 0 else "fail"
            md += [f"- surrogate-50 {s50['success_auc']:.4f}; BPDA-10 {b10['success_auc']:.4f}",
                   f"- **J3 / H6 worst case per video**: {w.mean():.4f} vs undefended {uw.mean():.4f}: {f(h6)} → **{status['J3'].upper()}**"]
            J["H6"] = h6
    for k in ("J1", "J2", "J3"):
        if status.get(k) == "fail":
            md.append(f"\n**JPEG-85 branch CUT at {k}.**")
            break

    # ================= Q: int8 recipes
    md += ["", "## Q. int8 backbone recipes", "", "### Q0 DEV pilot (TEST videos never used, not even for calibration)", ""]
    dref = L(D + "fp32_clean_nodef_t4")
    f32 = latency("fp32")
    rec = {}
    if dref:
        md += ["| Recipe | Draw changes vs fp32 CPU (DEV) | Mean | Worst | Speed-up (4 thr) | Eligible |", "|---|---|---|---|---|---|"]
        for tag, (var, _) in RECIPES.items():
            runs = [L(D + f"{tag}_cal{k}") for k in (1, 2, 3)]
            lat = latency(var)
            if not all(runs) or lat is None or f32 is None:
                md.append(f"| {tag} | not complete | | | | |")
                continue
            ch = [float(np.mean(v(r, DEV) - v(dref, DEV))) for r in runs]
            sp = f32 / lat
            ok = np.mean(ch) >= -CLEAN_COST_MAX and min(ch) >= WORST_DRAW_MIN and sp >= SPEEDUP_MIN
            rec[tag] = dict(var=var, changes=ch, mean=float(np.mean(ch)), worst=float(min(ch)), speedup=sp, eligible=bool(ok))
            md.append(f"| {tag} | {', '.join(f'{c:+.4f}' for c in ch)} | {np.mean(ch):+.4f} | {min(ch):+.4f} | {sp:.2f}× | {'yes' if ok else 'no'} |")
    if len(rec) == len(RECIPES) and not SELF.exists():
        elig = [t for t in rec if rec[t]["eligible"]]
        if elig:
            # best mean accuracy; within 0.002 of the best, prefer the faster recipe
            best = max(rec[t]["mean"] for t in elig)
            near = [t for t in elig if rec[t]["mean"] >= best - 0.002]
            pick = max(near, key=lambda t: (round(rec[t]["speedup"], 1), rec[t]["mean"]))  # faster; then more accurate
            SELF.write_text(json.dumps(dict(recipe=pick, args=f"--variant {rec[pick]['var']} {RECIPES[pick][1]}",
                                            table=rec, chosen_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
        else:
            SELF.write_text(json.dumps(dict(recipe=None, args="", table=rec, reason="no recipe eligible on DEV"), indent=1))
    sel = json.loads(SELF.read_text()) if SELF.exists() else None
    if sel is None:
        status["Q0"] = "run"
    elif sel["recipe"] is None:
        status["Q0"] = "fail"
        md.append("\n**No recipe met the DEV rule: int8 branch CUT (and the combined branch).**")
    else:
        status["Q0"] = "pass"
        md.append(f"\nChosen on DEV: **{sel['recipe']}** (`{sel['args']}`)")
    if status.get("Q0") == "pass":
        q = [L(T + f"int8sel_clean_nodef_t4_cal{k}") for k in (1, 2, 3)]
        if not all(q):
            status["Q1"] = "run"
        else:
            ch = [boot(v(r, S40) - v(cpu123, S40)) for r in q]
            m = float(np.mean([c["mean"] for c in ch]))
            sp = sel["table"][sel["recipe"]]["speedup"]
            status["Q1"] = "pass" if m >= -CLEAN_COST_MAX and sp >= SPEEDUP_MIN else "fail"
            md += ["", "### Q1 TEST, 3 calibration draws (rule H9')", ""] + \
                  [f"- draw {k}: {r['success_auc']:.4f}, change vs fp32 CPU {f(c)}" for k, (r, c) in enumerate(zip(q, ch), 1)] + \
                  [f"- **mean {m:+.4f}, SD {np.std([c['mean'] for c in ch], ddof=1):.4f}; speed-up {sp:.2f}× → {status['Q1'].upper()}**"]
            J["H9"] = dict(mean=m, draws=[c["mean"] for c in ch], speedup=sp)
            if status["Q1"] == "fail":
                md.append("\n**int8 branch CUT at Q1.**")

    # ================= C: combined
    md += ["", "## C. int8 (chosen recipe) + JPEG-85 on CPU", ""]
    j_cut = any(status.get(k) == "fail" for k in ("J1", "J2", "J3"))
    if status.get("J1") == "pass" and status.get("Q1") == "pass" and not j_cut:
        c1 = L(T + "int8sel_clean_jpeg85_t4")
        if not c1:
            status["C1"] = "run"
        else:
            g = boot(v(c1, S40) - v(cpu123, S40))
            lat = latency(sel["table"][sel["recipe"]]["var"])
            sp = f32 / (lat + c1["ms_per_frame"]["defense"]) if (lat and f32) else None
            status["C1"] = "pass" if g["mean"] >= -CLEAN_COST_MAX and sp and sp >= SPEEDUP_MIN else "fail"
            md.append(f"- **C1 clean**: {c1['success_auc']:.4f}, change vs fp32 CPU {f(g)}; speed incl. JPEG {sp:.2f}× → **{status['C1'].upper()}**")
            J["C1"] = dict(change=g, speedup=sp)
        if status.get("C1") == "pass":
            n0, b, s = (L(T + x) for x in ("int8sel_pgd4s10_nodef_t4", "int8sel_pgd4s10_jpeg85_adaptive_t4",
                                           "int8sel_pgd4s10_jpeg85_adaptive_diff_t4"))
            if not (n0 and b and s):
                status["C2"] = "run"
            else:
                w = np.minimum(v(b, S40), v(s, S40))
                g = boot(w - v(n0, S40))
                status["C2"] = "pass" if g["mean"] >= GAIN_MIN and g["lo"] > 0 else "fail"
                md.append(f"- **C2 attacks**: undefended {n0['success_auc']:.4f}; BPDA-10 {b['success_auc']:.4f}; surrogate-10 "
                          f"{s['success_auc']:.4f}; worst-case gain {f(g)} → **{status['C2'].upper()}**")
    else:
        cut = j_cut or any(status.get(k) == "fail" for k in ("Q0", "Q1"))
        md.append("- **CUT** (a JPEG or int8 gate failed)" if cut else "- waits for J1 and Q1 to pass")
        if cut:
            status["C1"] = status.get("C1") or "fail"   # E+R is settled once a prerequisite fails

    # ================= claims
    def P(*k):
        if any(status.get(x) == "fail" for x in k):
            return False                      # a required gate failed: the claim is settled
        return None if any(status.get(x) in (None, "run") for x in k) else True
    claims = {"Claim R-CPU (robust HiFT on CPU-class hardware): H6 + H7 + H8-CPU": P("J1", "J2", "J3", "H8cpu"),
              "Claim E (efficient HiFT): H9' + speed ≥ 1.5×": P("Q0", "Q1"),
              "Claim E+R (efficient and robust): all of the above + C1 + C2": P("J1", "J2", "J3", "H8cpu", "Q0", "Q1", "C1", "C2")}
    md += ["", "## Claims", ""]
    for k, x in claims.items():
        md.append(f"- {k}: **{'IMPROVEMENT' if x else 'not met' if x is False else 'pending'}**")
    J["claims"], J["status"] = claims, status

    # ================= what to submit next
    nxt = []
    j_dead = any(status.get(k) == "fail" for k in ("J1", "J2", "J3"))
    want = {"J1": status.get("J1") == "run", "J1c": not j_dead,   # J1c only serves the JPEG claim "Q0": status.get("Q0") == "run" or not dref, "Q0b": True,
            "J2": status.get("J2") == "run", "J3": status.get("J3") == "run", "Q1": status.get("Q1") == "run",
            "C1": status.get("C1") == "run", "C2": status.get("C2") == "run"}
    for step, on in want.items():
        for q in sorted(QDIR.glob("*.tsv")):
            if not (q.name.split("_")[0] == step or q.name.startswith(step + "-")):
                continue
            if on and not queue_done(q):
                nxt.append(str(q.relative_to(ROOT)))
    md += ["", "## NEXT (submit these, GPU/mix one at a time; CPU queues may run beside a GPU job)", ""]
    md += [f"- `{q}`" for q in nxt] or ["- nothing: round 4 is finished (or every branch is cut). Package the results."]
    J["next"] = nxt
    (OUT / "criteria.md").write_text("\n".join(md) + "\n")
    (OUT / "criteria.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
