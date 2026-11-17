"""Stage-4 preregistered decision (reports/STAGE4_PREREGISTRATION.md): S4-GAP arms vs the REF-C control on V.
Inputs: in-job validation of the retrieved runs (artifacts/tc2-runs/<RUN>/validation.json and, for the mechanism
check, validation/V1__*.csv predictions + the V1 annotations in artifacts/stage3-diag). Selection on V only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"artifacts/tc2-runs"
V1_GT = ROOT/"artifacts/stage3-diag/x/data/visdrone/SOT/VisDrone2019-SOT-val/VisDrone2019-SOT-val/annotations"
SEEDS = ("", "-S2")
WINDOWS = ((0, 100), (100, 300), (300, None))


def run_dir(base, seed):
    found = sorted(RUNS.glob(f"{base}{seed}-00[1-9]"))
    return found[-1] if found else None


def load(base):
    out = []
    for seed in SEEDS:
        d = run_dir(base, seed)
        if d is None or not (d/"validation.json").is_file():
            raise SystemExit(f"missing {base}{seed}")
        out.append((d, json.loads((d/"validation.json").read_text())))
    return out


ARMS = {"REF": "TC2-P1-REF", "REFC": "TC2-S4-REFC", "G300": "TC2-S4-G300", "GMIX": "TC2-S4-GMIX"}
data = {k: load(v) for k, v in ARMS.items()}
listing = {rec["episode_listing_sha256"] for runs in data.values() for _, rec in runs}
assert len(listing) == 1, listing
EPIS = data["REF"][0][1]["episodes"]
GROUP = {f'{e["subset"]}/{e["name"]}': e["group"] for e in EPIS}
STRATA = {"V1": [k for k in GROUP if k.startswith("V1/")],
          "V2veh": [k for k in GROUP if k.startswith("V2/") and GROUP[k] == "vehicle"],
          "V2per": [k for k in GROUP if k.startswith("V2/") and GROUP[k] == "person"],
          "V3": [k for k in GROUP if k.startswith("V3/")]}


def parts(rec, strata=STRATA):
    pe = rec["per_episode"]
    s = {name: np.mean([pe[k]["success_auc"] for k in keys]) for name, keys in strata.items()}
    s["nonV1"] = np.mean([np.mean([s["V2veh"], s["V2per"]]), s["V3"]])
    s["M"] = np.mean([s["V1"], np.mean([s["V2veh"], s["V2per"]]), s["V3"]])
    return s


for runs in data.values():
    for _, rec in runs:
        assert abs(parts(rec)["M"]-rec["summary"]["M"]) < 1e-9
avg = lambda arm, strata=STRATA: {k: np.mean([parts(rec, strata)[k] for _, rec in data[arm]]) for k in parts(data[arm][0][1], strata)}
seed_m = lambda arm, strata=STRATA: [parts(rec, strata)["M"] for _, rec in data[arm]]

ref, refc = avg("REF"), avg("REFC")
repro = [b-a for a, b in zip(seed_m("REF"), seed_m("REFC"))]
gate = abs(refc["M"]-ref["M"]) <= 0.003
print(f"reproducibility: REF M {ref['M']:.4f}, REF-C M {refc['M']:.4f} (seed-avg diff {refc['M']-ref['M']:+.4f}; "
      f"per seed {repro[0]:+.4f} {repro[1]:+.4f}) -> {'PASS' if gate else 'FAIL (no adoption; reproducibility finding)'}")


def iou(a, b):
    w = np.clip(np.minimum(a[:, 0]+a[:, 2], b[:, 0]+b[:, 2])-np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 1]+a[:, 3], b[:, 1]+b[:, 3])-np.maximum(a[:, 1], b[:, 1]), 0, None)
    return w*h/(a[:, 2]*a[:, 3]+b[:, 2]*b[:, 3]-w*h)


def windows(run):  # V1 sequence-mean IoU per frame window (valid GT frames only)
    out = [[] for _ in WINDOWS]
    for f in sorted(V1_GT.glob("*.txt")):
        g = np.loadtxt(f, delimiter=",", ndmin=2)
        p = np.loadtxt(run/"validation"/f"V1__{f.stem}.csv", delimiter=",", ndmin=2)
        ok, i, v = (g[:, 2] > 0) & (g[:, 3] > 0), np.arange(len(g)), iou(g, p)
        for k, (lo, hi) in enumerate(WINDOWS):
            m = ok & (i >= lo) & (i < (hi or len(g)))
            if m.any():
                out[k].append(v[m].mean())
    return [float(np.mean(x)) for x in out]


rng = np.random.default_rng(20261102)
draws = [{name: rng.integers(0, len(keys), len(keys)) for name, keys in STRATA.items()} for _ in range(2000)]
result = {"reproducibility": {"gate": gate, "ref_M": ref["M"], "refc_M": refc["M"], "per_seed": repro}, "arms": {}}
for arm in ("G300", "GMIX"):
    a = avg(arm)
    d = {k: a[k]-refc[k] for k in a}
    seeds = [x-y for x, y in zip(seed_m(arm), seed_m("REFC"))]
    loo = {k: avg(arm, dict(STRATA, V1=[x for x in STRATA["V1"] if x != k]))["M"]
           - avg("REFC", dict(STRATA, V1=[x for x in STRATA["V1"] if x != k]))["M"] for k in STRATA["V1"]}
    boot = [avg(arm, {n: [STRATA[n][i] for i in idx] for n, idx in b.items()})["M"]
            - avg("REFC", {n: [STRATA[n][i] for i in idx] for n, idx in b.items()})["M"] for b in draws]
    crit = {"1 seed-avg dM >= +0.015": d["M"] >= 0.015, "2 each seed dM >= +0.010": min(seeds) >= 0.010,
            "3 no subset drop > 0.015": min(d["V1"], d["V2veh"], d["V2per"], d["V3"]) >= -0.015,
            "4 non-V1 dM >= -0.005": d["nonV1"] >= -0.005, "5 LOO-V1 dM >= +0.008": min(loo.values()) >= 0.008,
            "6 bootstrap P10 dM > 0": float(np.percentile(boot, 10)) > 0}
    try:
        w = [[x-y for x, y in zip(windows(ra), windows(rc))] for (ra, _), (rc, _) in zip(data[arm], data["REFC"])]
        mech = {"dIoU_by_window_per_seed": w, "late_up_both_seeds": all(s[2] > 0 for s in w),
                "early_drop_lt_0.01": float(np.mean([s[0] for s in w])) > -0.01}
        mech["holds"] = mech["late_up_both_seeds"] and mech["early_drop_lt_0.01"]
    except OSError as exc:
        mech = {"unavailable": str(exc)}
    adopt = gate and all(crit.values())
    result["arms"][arm] = {"dM": d, "seed_dM": seeds, "loo_v1_min": min(loo.values()), "boot_p10": float(np.percentile(boot, 10)),
                           "criteria": crit, "qualifies": adopt, "mechanism": mech}
    print(f"\n{arm}: M {a['M']:.4f} dM {d['M']:+.4f} seeds {seeds[0]:+.4f}/{seeds[1]:+.4f}; "
          + " ".join(f"d{k} {d[k]:+.4f}" for k in ("V1", "V2veh", "V2per", "V3", "nonV1")))
    print(f"  LOO-V1 min {min(loo.values()):+.4f}; bootstrap P10 {np.percentile(boot, 10):+.4f}")
    for k, ok in crit.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"  mechanism (V1 IoU by frame window [0,100) [100,300) [300,end), arm - REF-C): {mech}")
ok = {arm: r for arm, r in result["arms"].items() if r["qualifies"]}
result["adopted"] = max(ok, key=lambda arm: ok[arm]["dM"]["M"]) if ok else None
print("\nDECISION:", f"ADOPT {result['adopted']} (open the sealed test once: adopted arm + REF-C)" if ok
      else "REJECT both arms (sealed test stays closed)")
(ROOT/"artifacts/stage4").mkdir(parents=True, exist_ok=True)
json.dump(result, open(ROOT/"artifacts/stage4/stage4_decision.json", "w"), indent=1, default=float)
