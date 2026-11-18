"""Stage-3 preregistered decision (reports/STAGE3_PREREGISTRATION.md) from the retrieved grid outputs.
Selection on V only; per-episode AUCs are sequence-macro, so subset AUC = mean of per-episode AUC."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT/"artifacts/tc2-validation"
MODELS = {"s1": "S3GRID-TC2-P1-REF-001", "s2": "S3GRID-TC2-P1-REF-S2-001", "b0": "S3GRID-B0-SIGMOID"}
grid = {}
listing = set()
for key, d in MODELS.items():
    for f in sorted((V/d).glob("*/validation.json")):
        rec = json.loads(f.read_text())
        listing.add(rec["episode_listing_sha256"])
        grid.setdefault(rec["grid_label"], {})[key] = rec
assert len(listing) == 1, listing
assert all(len(v) == 3 for v in grid.values()), "missing model x config"
EPIS = grid["upstream"]["s1"]["episodes"]
GROUP = {f'{e["subset"]}/{e["name"]}': e["group"] for e in EPIS}


def video(key):
    sub, name = key.split("/", 1)
    if sub == "V1":
        return "V1:"+name                      # 11 distinct SOT sequences
    if sub == "V2":
        return "V2:"+name.split("#")[0]        # MOT video
    return "V3:"+name.split("#")[0].split("_")[0]  # UAVDT video


def parts(rec, keep=None):
    pe = {k: v["success_auc"] for k, v in rec["per_episode"].items() if keep is None or keep(k)}
    sel = lambda sub, grp=None: [a for k, a in pe.items() if k.startswith(sub+"/") and (grp is None or GROUP[k] == grp)]
    v1, veh, per, v3 = (np.mean(x) if x else np.nan for x in (sel("V1"), sel("V2", "vehicle"), sel("V2", "person"), sel("V3")))
    return {"V1": v1, "V2veh": veh, "V2per": per, "V3": v3,
            "nonV1": np.nanmean([np.nanmean([veh, per]), v3]), "M": np.nanmean([v1, np.nanmean([veh, per]), v3])}


# sanity: recomputed M equals the recorded summary M
for lab, recs in grid.items():
    for key, rec in recs.items():
        assert abs(parts(rec)["M"]-rec["summary"]["M"]) < 1e-9, (lab, key)

seed_avg = lambda lab, keep=None: {k: np.mean([parts(grid[lab][s], keep)[k] for s in ("s1", "s2")]) for k in
                                   ("V1", "V2veh", "V2per", "V3", "nonV1", "M")}
ranked = sorted(grid, key=lambda lab: -seed_avg(lab)["M"])
up = seed_avg("upstream")
print(f"upstream REF seed-avg: " + " ".join(f"{k} {v:.4f}" for k, v in up.items()))
print("top 8 by seed-averaged M (dM vs upstream; per-seed dM):")
for lab in ranked[:8]:
    a = seed_avg(lab)
    ds = [parts(grid[lab][s])["M"]-parts(grid["upstream"][s])["M"] for s in ("s1", "s2")]
    print(f"  {lab:34s} M {a['M']:.4f} dM {a['M']-up['M']:+.4f}  seeds {ds[0]:+.4f} {ds[1]:+.4f}")
print(f"  ... worst: {ranked[-1]} dM {seed_avg(ranked[-1])['M']-up['M']:+.4f}")

c = ranked[0]
a = seed_avg(c)
d = {k: a[k]-up[k] for k in a}
seeds = [parts(grid[c][s])["M"]-parts(grid["upstream"][s])["M"] for s in ("s1", "s2")]
v1names = [k for k in GROUP if k.startswith("V1/")]
loo = {k.split("/")[1]: seed_avg(c, lambda x, k=k: x != k)["M"]-seed_avg("upstream", lambda x, k=k: x != k)["M"]
       for k in v1names}
crit = {
    "1 seed-avg dM >= +0.015": d["M"] >= 0.015,
    "2 each seed dM >= +0.010": min(seeds) >= 0.010,
    "3 no subset drop > 0.015": min(d["V1"], d["V2veh"], d["V2per"], d["V3"]) >= -0.015,
    "4 non-V1 part not lower": d["nonV1"] >= 0,
    "5 LOO-V1 dM >= +0.008": min(loo.values()) >= 0.008,
}
print(f"\nc* = {c}: dM {d['M']:+.4f} seeds {seeds[0]:+.4f}/{seeds[1]:+.4f}; "
      + " ".join(f"d{k} {d[k]:+.4f}" for k in ("V1", "V2veh", "V2per", "V3", "nonV1")))
print(f"  LOO-V1 min dM {min(loo.values()):+.4f} (drop {min(loo, key=loo.get)}), max {max(loo.values()):+.4f}")
for k, ok in crit.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
adopt = all(crit.values())
print("DECISION:", "ADOPT REF/T = "+c if adopt else "REJECT (sealed test stays closed)")

# optimism estimate: two-fold split by video, select on one half, score on the other (seed-averaged), 20 splits
vids = sorted({video(k) for k in GROUP})
rng = random.Random(20261102)
gaps, held = [], []
for _ in range(20):
    rng.shuffle(vids)
    half = set(vids[:len(vids)//2])
    for fold in (lambda k: video(k) in half, lambda k: video(k) not in half):
        other = lambda k, f=fold: not f(k)
        upf, upo = seed_avg("upstream", fold)["M"], seed_avg("upstream", other)["M"]
        best = max(grid, key=lambda lab: seed_avg(lab, fold)["M"])
        gaps.append((seed_avg(best, fold)["M"]-upf)-(seed_avg(best, other)["M"]-upo))
        held.append(seed_avg(best, other)["M"]-upo)
print(f"\ntwo-fold by video (20 random splits): in-fold minus held-out dM (optimism) mean {np.mean(gaps):+.4f}; "
      f"held-out dM of the fold-selected config mean {np.mean(held):+.4f} (min {min(held):+.4f})")

b0u, b0c = parts(grid["upstream"]["b0"]), parts(grid[c]["b0"])
print(f"B0-sigmoid: upstream M {b0u['M']:.4f}, c* M {b0c['M']:.4f} (dM {b0c['M']-b0u['M']:+.4f}); "
      + " ".join(f"d{k} {b0c[k]-b0u[k]:+.4f}" for k in ("V1", "V2veh", "V2per", "V3")))
b0best = max(grid, key=lambda lab: parts(grid[lab]["b0"])["M"])
print(f"B0 own best config: {b0best} M {parts(grid[b0best]['b0'])['M']:.4f}")
json.dump({"c_star": c, "adopt": adopt, "criteria": crit, "dM": d, "seed_dM": seeds, "loo_v1": loo,
           "optimism_mean": float(np.mean(gaps)), "heldout_dM_mean": float(np.mean(held)),
           "ranked": [(lab, seed_avg(lab)["M"]) for lab in ranked],
           "b0": {"upstream": b0u, "c_star": b0c, "own_best": b0best}},
          open(ROOT/"artifacts/stage3-diag/stage3_decision.json", "w"), indent=1, default=float)
