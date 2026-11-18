"""Stage-5 preregistered decision (reports/STAGE5_PREREGISTRATION.md): REF + VisDrone-SOT train_part2 (SOT2) vs REF
on V. REF is its own control: stage 4 showed REF retraining is bitwise reproducible (identical final-model.pt).
Inputs: artifacts/tc2-runs/<RUN>/validation.json (+ validation/V1__*.csv for the descriptive V1 frame windows)
and the part2 leakage report artifacts/stage5/SOT2-overlap.json. Selection on V only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"artifacts/tc2-runs"
V1_GT = ROOT/"artifacts/stage3-diag/x/data/visdrone/SOT/VisDrone2019-SOT-val/VisDrone2019-SOT-val/annotations"
SEEDS = ("", "-S2")
WINDOWS = ((0, 100), (100, 300), (300, None))
ARMS = {"REF": "TC2-P1-REF", "SOT2": "TC2-S5-SOT2"}


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


leak = json.loads((ROOT/"artifacts/stage5/SOT2-overlap.json").read_text())
dups = {name: rec["duplicate_train_frames"] for name, rec in leak["eval"].items()}
leak_ok = set(dups) >= {"V1", "V2"} and not any(dups.values())
print(f"leakage (byte-identical part2 frames): {dups} -> {'PASS' if leak_ok else 'FAIL (no adoption; investigate)'}")

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


ref, a = avg("REF"), avg("SOT2")
d = {k: a[k]-ref[k] for k in a}
seeds = [x-y for x, y in zip(seed_m("SOT2"), seed_m("REF"))]
loo = {k: avg("SOT2", dict(STRATA, V1=[x for x in STRATA["V1"] if x != k]))["M"]
       - avg("REF", dict(STRATA, V1=[x for x in STRATA["V1"] if x != k]))["M"] for k in STRATA["V1"]}
rng = np.random.default_rng(20261103)
draws = [{name: rng.integers(0, len(keys), len(keys)) for name, keys in STRATA.items()} for _ in range(2000)]
boot = [avg("SOT2", {n: [STRATA[n][i] for i in idx] for n, idx in b.items()})["M"]
        - avg("REF", {n: [STRATA[n][i] for i in idx] for n, idx in b.items()})["M"] for b in draws]
crit = {"1 seed-avg dM >= +0.010": d["M"] >= 0.010, "2 each seed dM >= +0.005": min(seeds) >= 0.005,
        "3 no subset drop > 0.015": min(d["V1"], d["V2veh"], d["V2per"], d["V3"]) >= -0.015,
        "4 non-V1 dM >= -0.005": d["nonV1"] >= -0.005, "5 LOO-V1 dM >= +0.005": min(loo.values()) >= 0.005,
        "6 bootstrap P10 dM > 0": float(np.percentile(boot, 10)) > 0}
try:  # REF frame predictions come from REF-C (bitwise-identical weights, stage 4); REF's own CSVs stayed on TC2
    refc = [run_dir("TC2-S4-REFC", seed) for seed in SEEDS]
    w = [[x-y for x, y in zip(windows(ra), windows(rr))] for (ra, _), rr in zip(data["SOT2"], refc)]
except OSError as exc:
    w = f"unavailable: {exc}"
adopt = leak_ok and all(crit.values())
print(f"\nSOT2: M {a['M']:.4f} vs REF {ref['M']:.4f}; dM {d['M']:+.4f} seeds {seeds[0]:+.4f}/{seeds[1]:+.4f}; "
      + " ".join(f"d{k} {d[k]:+.4f}" for k in ("V1", "V2veh", "V2per", "V3", "nonV1")))
print(f"  LOO-V1 min {min(loo.values()):+.4f}; bootstrap P10 {np.percentile(boot, 10):+.4f}")
for k, ok in crit.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
print(f"  descriptive: V1 dIoU by frame window [0,100) [100,300) [300,end), per seed (SOT2 - REF): {w}")
print("\nDECISION:", "ADOPT SOT2 (local test-dev leakage check, then open the sealed test once for SOT2 seeds 1-2)"
      if adopt else "REJECT SOT2 (sealed test stays closed)")
result = {"leakage": {"duplicates": dups, "ok": leak_ok}, "dM": d, "seed_dM": seeds, "loo_v1_min": min(loo.values()),
          "boot_p10": float(np.percentile(boot, 10)), "criteria": crit, "v1_windows_dIoU": w, "adopted": adopt}
(ROOT/"artifacts/stage5").mkdir(parents=True, exist_ok=True)
json.dump(result, open(ROOT/"artifacts/stage5/stage5_decision.json", "w"), indent=1, default=float)
