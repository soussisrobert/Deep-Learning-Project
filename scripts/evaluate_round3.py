"""Score round 3 against the pre-registered rules in ROUND3_PLAN.md.

  python scripts/evaluate_round3.py --results results --out results/round3

Reads finished runs only ("not run yet" otherwise, never guessed). Paired by video, 95% bootstrap
over videos (10,000 resamples). Thresholds are the round-2 ones, unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CLEAN_COST_MAX = 0.010   # HiFT's lead over SiamRPN++ (0.589 vs 0.579)
GAIN_MIN = 0.10          # success regained under the adaptive attack
OVERHEAD_MAX = 0.0385    # 1 - 30/31.2: HiFT's onboard real-time margin
SPEEDUP_MIN = 1.5
SEED_SD_MAX = 0.02
T = "uav123-sub40_"


class Res:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, name):
        d = self.root / name
        if not (d / "summary.json").exists():
            return None
        s = json.loads((d / "summary.json").read_text())
        s["per"] = {p["name"]: p["success_auc"] for p in json.loads((d / "per_seq.json").read_text())}
        return s


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), better=int((d > .05).sum()),
                worse=int((d < -.05).sum()))


def per(s, keys):
    return np.array([s["per"][k] for k in keys])


def paired(a, b, keys):
    return boot(per(a, keys) - per(b, keys))


def worst(runs, keys):
    """Per-video minimum over attacks: the attacker may pick the best attack for each video."""
    return np.min(np.stack([per(r, keys) for r in runs]), 0)


def fmt(p):
    return f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] (+{p['better']}/−{p['worse']})" if p else "–"


def a(s):
    return f"{s['success_auc']:.3f}" if s else "not run yet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/round3")
    ap.add_argument("--subset", default="attack-subset-40.json")
    o = ap.parse_args()
    R, out = Res(o.results), Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    S = sorted(json.loads(Path(o.subset).read_text())["sequences"])
    sel_f = Path("plan/round3/selected.json")
    sel = json.loads(sel_f.read_text()) if sel_f.exists() else None
    md, J = ["# Round 3: results against the pre-registered rules (ROUND3_PLAN.md)", ""], {}
    clean, pgd10, pgd50 = R(T + "fp32_clean_nodef"), R(T + "fp32_pgd4s10_nodef"), R(T + "fp32_pgd4s50_nodef")
    if not (clean and pgd10):
        raise SystemExit("round-2 baselines missing")
    und = [x for x in (pgd10, pgd50) if x]
    und_worst = worst(und, S)
    md += [f"TEST = fixed 40 videos. Clean {clean['success_auc']:.3f}. Undefended PGD-10 {pgd10['success_auc']:.3f}, "
           f"PGD-50 {a(pgd50)}, worst per video {und_worst.mean():.3f}.", ""]

    # ---------- A. does round-2 JPEG75 survive stronger attackers?
    md += ["## A. JPEG75 (round 2) under stronger adaptive attacks", "",
           "| Attack on JPEG75 | Success AUC | Gain vs undefended (same strength) |", "|---|---|---|"]
    atk75 = [("BPDA, 10 steps (round 2)", "fp32_pgd4s10_jpeg75_adaptive", pgd10),
             ("surrogate gradient, 10 steps", "fp32_pgd4s10_jpeg75_adaptive_diff", pgd10),
             ("surrogate gradient, 50 steps", "fp32_pgd4s50_jpeg75_adaptive_diff", pgd50),
             ("BPDA, 50 steps", "fp32_pgd4s50_jpeg75_adaptive", pgd50)]
    got75 = []
    for lab, n, ref in atk75:
        s = R(T + n)
        md.append(f"| {lab} | {a(s)} | {fmt(paired(s, ref, S)) if (s and ref) else '–'} |")
        if s:
            got75.append(s)
    if got75:
        w = worst(got75, S)
        g = boot(w - und_worst)
        md.append(f"| **worst per video over the {len(got75)} attacks** | {w.mean():.3f} | {fmt(g)} vs worst undefended |")
        J["jpeg75_worst_case_gain"] = g

    # ---------- B. selected quality
    md += ["", "## B. Selected JPEG quality on TEST (rules H6, H7)", ""]
    if not sel:
        md.append("Selection not made yet (run scripts/r3_select.py after stage 1).")
    else:
        md.append(f"Chosen on DEV only: **{sel['jpeg']}** ({sel['reason']}).")
        md += ["", "| Quality | DEV clean cost | DEV robust AUC | eligible |", "|---|---|---|---|"]
        md += [f"| {r['quality']} | {r['clean_cost']:+.4f} | {r['robust_auc']:.3f} | {'yes' if r['eligible'] else 'no'} |"
               for r in sel["table"]]
        c = R(T + "fp32_clean_jpegsel")
        pc = paired(c, clean, S) if c else None
        atks = [("BPDA-10", "fp32_pgd4s10_jpegsel_adaptive", pgd10), ("surrogate-10", "fp32_pgd4s10_jpegsel_adaptive_diff", pgd10),
                ("surrogate-50", "fp32_pgd4s50_jpegsel_adaptive_diff", pgd50)]
        md += ["", f"Clean: {a(c)} — change {fmt(pc)} → H7 (≥ −{CLEAN_COST_MAX:.3f}): "
               + ("not run yet" if not pc else ("PASS" if pc["mean"] >= -CLEAN_COST_MAX else "FAIL")), "",
               "| Attack on the selected JPEG | Success AUC | Gain vs undefended (same strength) |", "|---|---|---|"]
        got = []
        for lab, n, ref in atks:
            s = R(T + n)
            md.append(f"| {lab} | {a(s)} | {fmt(paired(s, ref, S)) if (s and ref) else '–'} |")
            if s:
                got.append(s)
        h6 = None
        if len(got) == len(atks) and pgd50:
            w = worst(got, S)
            h6 = boot(w - und_worst)
            md.append(f"| **worst per video (all 3)** | {w.mean():.3f} | {fmt(h6)} vs worst undefended |")
        md.append("")
        md.append("H6 (worst-case gain ≥ +0.10, interval above 0): " +
                  ("not run yet" if not h6 else ("PASS" if h6["mean"] >= GAIN_MIN and h6["lo"] > 0 else "FAIL")))
        seeds = [R(T + "fp32_pgd4s10_jpegsel_adaptive_diff" + x) for x in ("", "_seed1", "_seed2")]
        if all(seeds):
            v = [x["success_auc"] for x in seeds]
            md.append(f"\nSeeds (surrogate-10): {', '.join(f'{x:.3f}' for x in v)} → SD {np.std(v, ddof=1):.3f} "
                      f"({'stable' if np.std(v, ddof=1) <= SEED_SD_MAX else 'UNSTABLE: report mean ± SD'})")
        tc, ta = R(T + "fp32_clean_tjpegsel"), R(T + "fp32_pgd4s10_tjpegsel_adaptive_diff")
        md.append(f"\nGPU JPEG ({sel['tjpeg']}): clean {a(tc)} ({fmt(paired(tc, clean, S)) if tc else '–'}); "
                  f"surrogate-10 {a(ta)} ({fmt(paired(ta, pgd10, S)) if ta else '–'}).")
        J["selected"] = dict(sel=sel, clean_change=pc, h6=h6)

    # ---------- C. overhead
    md += ["", "## C. Time overhead (rule H8: ≤ 3.85% of one tracking step on the same device)", "",
           "| Defence | Device | Defence ms / frame | Tracking ms / frame (no defence) | Share | H8 |", "|---|---|---|---|---|---|"]
    ovh = {}
    rows = [("jpeg75 (round 2)", R(T + "fp32_clean_jpeg75"), clean, "GPU (A40)")]
    if sel:
        rows += [(sel["jpeg"], R(T + "fp32_clean_jpegsel"), clean, "GPU (A40)"),
                 (sel["tjpeg"], R(T + "fp32_clean_tjpegsel"), clean, "GPU (A40), JPEG on GPU"),
                 (sel["jpeg"] + " + int8bb", R(T + "int8bb_clean_jpegsel_t4"), R(T + "int8bb_clean_nodef_t4"), "CPU, 4 threads")]
    for lab, s, ref, dev in rows:
        if s and ref:
            sh = s["ms_per_frame"]["defense"] / ref["ms_per_frame"]["total"]
            ovh[lab] = sh
            md.append(f"| {lab} | {dev} | {s['ms_per_frame']['defense']:.2f} | {ref['ms_per_frame']['total']:.2f} | {sh:.1%} | "
                      f"{'PASS' if sh <= OVERHEAD_MAX else 'FAIL'} |")
        else:
            md.append(f"| {lab} | {dev} | not run yet | | | |")
    for f in sorted(Path(o.results, "bench").glob("*/summary.json")):
        b = json.loads(f.read_text())
        md.append(f"\nBenchmark {b['bench']}: " + "; ".join(f"{r['defense']} {r['defense_ms']:.2f} ms ({r['share_of_model']:.1%} of network)"
                                                         for r in b["rows"]))
    J["overhead"] = ovh

    # ---------- D. int8 calibration spread
    md += ["", "## D. int8 backbone: calibration spread (rule H9: mean change ≥ −0.010)", ""]
    cpu = R("uav123_fp32_clean_nodef_t4")
    cals = [("round-2 set", R(T + "int8bb_clean_nodef_t4"))] + [(f"set {k}", R(T + f"int8bb_clean_nodef_t4_cal{k}")) for k in (1, 2, 3)]
    md += ["| Calibration | Success AUC (40) | Change vs fp32 CPU |", "|---|---|---|"]
    ch = []
    for lab, s in cals:
        if s and cpu:
            p = paired(s, cpu, S)
            ch.append(p["mean"])
            md.append(f"| {lab} | {s['success_auc']:.3f} | {fmt(p)} |")
        else:
            md.append(f"| {lab} | not run yet | |")
    if len(ch) == 4:
        h9 = float(np.mean(ch))
        md.append(f"\nMean over 4 calibration sets {h9:+.4f}, SD {np.std(ch, ddof=1):.4f} → H9 "
                  + ("PASS" if h9 >= -CLEAN_COST_MAX else "FAIL"))
        J["h9"] = dict(mean=h9, sd=float(np.std(ch, ddof=1)), each=ch)

    # ---------- E. combined int8 + JPEG on CPU
    md += ["", "## E. int8 backbone + selected JPEG on CPU (rule H10)", ""]
    if sel:
        cc, cb, cd = R(T + "int8bb_clean_jpegsel_t4"), R(T + "int8bb_pgd4s10_jpegsel_adaptive_t4"), R(T + "int8bb_pgd4s10_jpegsel_adaptive_diff_t4")
        ip = R(T + "int8bb_pgd4s10_nodef_t4")
        e = {}
        for sub in ("edge", "edge_round2"):
            for f in sorted(Path(o.results, sub).glob("cpu_t4_*.json")):
                j = json.loads(f.read_text())
                e[j["variant"]] = j["latency_ms_mean"]
        checks = {}
        if cc and cpu:
            p = paired(cc, cpu, S)
            checks["clean"] = p["mean"] >= -CLEAN_COST_MAX
            md.append(f"- Clean: {cc['success_auc']:.3f}, change vs fp32 CPU {fmt(p)} → "
                      f"{'PASS' if checks['clean'] else 'FAIL'} (≥ −0.010)")
            if "fp32" in e and "int8bb" in e:
                sp = e["fp32"] / (e["int8bb"] + cc["ms_per_frame"]["defense"])
                checks["speed"] = sp >= SPEEDUP_MIN
                md.append(f"- Speed: fp32 network {e['fp32']:.1f} ms vs int8bb {e['int8bb']:.1f} ms + JPEG "
                          f"{cc['ms_per_frame']['defense']:.2f} ms → {sp:.2f}× → {'PASS' if checks['speed'] else 'FAIL'} (≥ 1.5×)")
        got = [x for x in (cb, cd) if x]
        if got and ip:
            w = worst(got, S)
            g = boot(w - per(ip, S))
            checks["robust"] = g["mean"] >= GAIN_MIN and g["lo"] > 0 and len(got) == 2
            md.append(f"- Under attack (worst of {len(got)}: BPDA-10, surrogate-10): {w.mean():.3f} vs int8bb undefended "
                      f"{ip['success_auc']:.3f}: {fmt(g)} → {'PASS' if checks['robust'] else 'FAIL / incomplete'} (≥ +0.10)")
        J["h10"] = checks
    # ---------- F. corruptions
    md += ["", "## F. Natural corruptions (descriptive; severity 3, ImageNet-C levels)", "",
           "| Corruption | No defence (Δ vs clean) | With selected JPEG (Δ vs no defence) |", "|---|---|---|"]
    for cs in ("gauss_noise3", "gauss_blur3", "motion_blur3", "low_light3", "contrast3"):
        n0, n1 = R(T + f"fp32_{cs}_nodef"), R(T + f"fp32_{cs}_jpegsel")
        md.append(f"| {cs} | {a(n0)} ({fmt(paired(n0, clean, S)) if n0 else '–'}) | "
                  f"{a(n1)} ({fmt(paired(n1, n0, S)) if (n0 and n1) else '–'}) |")
    # ---------- verdicts
    md += ["", "## Verdicts", ""]
    sel_ok = J.get("selected", {})
    h6, h7 = sel_ok.get("h6"), sel_ok.get("clean_change")
    claims = {
        "Claim R (robust HiFT, GPU deployment): H6 + H7 + H8 on GPU (OpenCV or GPU JPEG)":
            None if not (h6 and h7 and sel) else (h6["mean"] >= GAIN_MIN and h6["lo"] > 0 and h7["mean"] >= -CLEAN_COST_MAX and
                                                  any(v <= OVERHEAD_MAX for k, v in ovh.items() if "int8" not in k and k != "jpeg75 (round 2)")),
        "Claim E+R (efficient and robust HiFT, CPU deployment): H6 + H9 + H10 (clean, speed, robust) + H8 on CPU":
            None if not (h6 and "h9" in J and J.get("h10") and sel) else (
                h6["mean"] >= GAIN_MIN and h6["lo"] > 0 and J["h9"]["mean"] >= -CLEAN_COST_MAX and
                all(J["h10"].get(k) for k in ("clean", "speed", "robust")) and
                any(v <= OVERHEAD_MAX for k, v in ovh.items() if "int8" in k)),
    }
    for k, v in claims.items():
        md.append(f"- {k}: **{'not all runs done' if v is None else ('IMPROVEMENT' if v else 'not met')}**")
    J["claims"] = {k: v for k, v in claims.items()}
    (out / "criteria.md").write_text("\n".join(md) + "\n")
    (out / "criteria.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
