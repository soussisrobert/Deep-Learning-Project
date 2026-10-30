"""Score round 2 against the pre-registered criteria in EXPERIMENT_PLAN.md.

  python scripts/evaluate_round2.py --results results --out results/round2

Reads finished runs only; anything missing is reported as "not run yet" (never guessed).
All comparisons are paired on the same sequences; uncertainty is a 95% bootstrap over
sequences (10,000 resamples), never over frames.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# ---- pre-registered thresholds (see EXPERIMENT_PLAN.md for where each number comes from)
CLEAN_COST_MAX = 0.010     # HiFT's own UAV123 lead over SiamRPN++: 0.589 vs 0.579 success
ADAPT_GAIN_MIN = 0.10      # success AUC regained under the adaptive attack
OVERHEAD_MAX = 0.0385      # HiFT onboard 31.2 FPS vs 30 FPS real time: 1 - 30/31.2
SPEEDUP_MIN = 1.5          # int8 literature reports ~1.9-2.2x (Jacob et al., CVPR 2018)
ACC_DROP_MAX = 0.010       # same yardstick as the clean cost
SEED_SD_MAX = 0.02
DEFENCES = ["blur1", "median3", "bits5", "jpeg75"]
P = "uav123-sub40_"


def load(res, name):
    d = Path(res) / name
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    s["per"] = {p["name"]: p for p in json.loads((d / "per_seq.json").read_text())}
    return s


def paired(a, b, seqs=None, n=10000, seed=0):
    """mean of (a - b) over common sequences, 95% bootstrap interval, #better, #worse."""
    common = sorted(set(a["per"]) & set(b["per"]) & (set(seqs) if seqs else set(a["per"])))
    d = np.array([a["per"][k]["success_auc"] - b["per"][k]["success_auc"] for k in common])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(common), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                better=int((d > 0.05).sum()), worse=int((d < -0.05).sum()))


def auc_on(s, seqs):
    return float(np.mean([np.mean(s["per"][k]["success_curve"]) for k in seqs]))


def fmt(p):
    return f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] (n={p['n']}, +{p['better']}/-{p['worse']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/round2")
    ap.add_argument("--subset", default="attack-subset-40.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S40 = json.loads(Path(a.subset).read_text())["sequences"]
    R = lambda n: load(a.results, n)  # noqa: E731
    md, J = ["# Round 2: results against the pre-registered criteria", ""], {}
    clean, pgd = R(P + "fp32_clean_nodef"), R(P + "fp32_pgd4s10_nodef")
    if not (clean and pgd):
        raise SystemExit("baseline runs (clean, PGD-10 eps 4 on the 40 sequences) are missing")
    lost = clean["success_auc"] - pgd["success_auc"]
    md += [f"Baseline on the 40 sequences: clean **{clean['success_auc']:.3f}**, PGD-10 ε=4 **{pgd['success_auc']:.3f}** "
           f"(lost {lost:.3f}). Clean tracking time {clean['ms_per_frame']['total']:.2f} ms/frame (no image loading).", ""]

    # ---------------- defences
    md += ["## 1. Defences (proposed improvement for robustness)", "",
           f"Pass = clean cost ≥ −{CLEAN_COST_MAX:.3f} AND adaptive gain ≥ +{ADAPT_GAIN_MIN:.2f} with interval above 0 "
           f"AND time overhead ≤ {OVERHEAD_MAX:.1%}.", "",
           "| Defence | Clean AUC (Δ vs clean) | PGD, attacker unaware (Δ vs PGD) | PGD, adaptive (Δ vs PGD) | Share of loss recovered (adaptive) | Overhead per frame | Verdict |",
           "|---|---|---|---|---|---|---|"]
    J["defences"] = {}
    for d in DEFENCES:
        c, u, ad = R(P + f"fp32_clean_{d}"), R(P + f"fp32_pgd4s10_{d}"), R(P + f"fp32_pgd4s10_{d}_adaptive")
        if not (c and u and ad):
            md.append(f"| {d} | " + " | ".join("not run yet" if x is None else f"{x['success_auc']:.3f}" for x in (c, u, ad))
                      + " | – | – | incomplete |")
            continue
        pc, pu, pa = paired(c, clean), paired(u, pgd), paired(ad, pgd)
        rec = pa["mean"] / lost if lost > 0 else float("nan")
        ovh_ms = c["ms_per_frame"]["total"] - clean["ms_per_frame"]["total"]
        ovh = ovh_ms / clean["ms_per_frame"]["total"]
        ok = [pc["mean"] >= -CLEAN_COST_MAX, pa["mean"] >= ADAPT_GAIN_MIN and pa["lo"] > 0, ovh <= OVERHEAD_MAX]
        verdict = "IMPROVEMENT" if all(ok) else ("robust but costly" if ok[1] else
                                                 ("helps only if attacker unaware" if pu["lo"] > 0 else "no gain"))
        md.append(f"| {d} | {c['success_auc']:.3f} ({fmt(pc)}) | {u['success_auc']:.3f} ({fmt(pu)}) | "
                  f"{ad['success_auc']:.3f} ({fmt(pa)}) | {rec:.0%} | {ovh_ms:+.2f} ms ({ovh:+.1%}) | {verdict} |")
        J["defences"][d] = dict(clean=c["success_auc"], unaware=u["success_auc"], adaptive=ad["success_auc"],
                                clean_cost=pc, gain_unaware=pu, gain_adaptive=pa, recovered=rec,
                                overhead_ms=ovh_ms, overhead=ovh, checks=ok, verdict=verdict)

    # ---------------- compression
    md += ["", "## 2. Compression (proposed improvement for efficiency)", "",
           f"Pass = network ≥ {SPEEDUP_MIN}× faster than fp32 at 4 CPU threads AND accuracy drop ≤ {ACC_DROP_MAX:.3f} "
           "(interval reported).", "",
           "| Variant | Network ms (4 thr) | Speed-up | Size MB | AUC on 40 (Δ vs fp32 CPU) | AUC on 123 (Δ vs fp32 CPU) | PGD ε=4, fp32 attacker (Δ vs fp32 PGD) | Verdict |",
           "|---|---|---|---|---|---|---|---|"]
    edge = {}
    for sub in ("edge", "edge_round2"):
        for f in sorted((Path(a.results) / sub).glob("cpu_t4_*.json")):
            j = json.loads(f.read_text())
            edge[j["variant"]] = j   # round 2 overrides round 1 for the same variant
    cpu_full = R("uav123_fp32_clean_nodef_t4")
    J["compression"] = {}
    for v in ["int8bb", "int8full", "prune50", "int8dyn"]:
        e, f32e = edge.get(v), edge.get("fp32")
        s40, s123, sp = R(P + f"{v}_clean_nodef_t4"), R(f"uav123_{v}_clean_nodef_t4"), R(P + f"{v}_pgd4s10_nodef_t4")
        if not e and not s40:
            continue
        spd = f32e["latency_ms_mean"] / e["latency_ms_mean"] if (e and f32e) else None
        p40 = paired(s40, cpu_full, S40) if (s40 and cpu_full) else None
        p123 = paired(s123, cpu_full) if (s123 and cpu_full) else None
        ppgd = paired(sp, pgd) if sp else None
        ok = [spd is not None and spd >= SPEEDUP_MIN, p40 is not None and p40["mean"] >= -ACC_DROP_MAX]
        verdict = ("IMPROVEMENT" if all(ok) else "faster but less accurate" if ok[0] and p40 else
                   "not faster" if spd is not None and not ok[0] else "incomplete")
        md.append(f"| {v} | {e['latency_ms_mean']:.1f} | {spd:.2f}× | {e['size_mb']:.1f} | "
                  if e else f"| {v} | – | – | – | ")
        md[-1] += (f"{s40['success_auc']:.3f} ({fmt(p40)}) | " if p40 else "not run yet | ")
        md[-1] += (f"{s123['success_auc']:.3f} ({fmt(p123)}) | " if p123 else "not run yet | ")
        md[-1] += (f"{sp['success_auc']:.3f} ({fmt(ppgd)}) | " if ppgd else "not run yet | ") + verdict + " |"
        J["compression"][v] = dict(speedup=spd, acc40=p40, acc123=p123, pgd=ppgd, checks=ok, verdict=verdict)
    if cpu_full:
        md.append(f"\nfp32 CPU reference on the same 40 sequences: {auc_on(cpu_full, S40):.3f} "
                  f"(all 123: {cpu_full['success_auc']:.3f}).")

    # ---------------- attack characterisation
    md += ["", "## 3. Attack strength and stability", "", "| Run | Success AUC | Δ vs clean |", "|---|---|---|"]
    J["attacks"] = {}
    for n in ["fp32_noise1_nodef", "fp32_fgsm1_nodef", "fp32_pgd1s10_nodef", "fp32_pgd2s10_nodef", "fp32_pgd4s10_nodef",
              "fp32_pgd8s10_nodef", "fp32_pgd4s10_nodef_seed1", "fp32_pgd4s10_nodef_seed2",
              "fp32_noise4_nodef", "fp32_noise4_nodef_seed1", "fp32_noise4_nodef_seed2"]:
        s = R(P + n)
        md.append(f"| {n} | " + (f"{s['success_auc']:.3f} | {fmt(paired(s, clean))} |" if s else "not run yet | – |"))
        if s:
            J["attacks"][n] = s["success_auc"]
    for kind in ("pgd4s10", "noise4"):
        vals = [J["attacks"].get(f"fp32_{kind}_nodef{suf}") for suf in ("", "_seed1", "_seed2")]
        if all(v is not None for v in vals):
            sd = float(np.std(vals, ddof=1))
            md.append(f"\n{kind} over 3 seeds: mean {np.mean(vals):.3f}, SD {sd:.3f} → "
                      + ("stable" if sd <= SEED_SD_MAX else "UNSTABLE: report the mean ± SD, not seed 0"))
    (out / "criteria.md").write_text("\n".join(md) + "\n")
    (out / "criteria.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
