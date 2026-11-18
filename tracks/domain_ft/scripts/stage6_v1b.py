"""Stage-6 preregistered V1B analysis (reports/STAGE6_PREREGISTRATION.md). V1B = 35 VisDrone-SOT train_part2 sequences
sharing no source video with training or the sealed test (configs/v1b_sequences.txt). Validation only, no adoption.
Inputs: artifacts/tc2-validation/V1B-<label>/validation.json (job hift_s6_v1b: subsets V1 + V1B) and the stored
stage 2-4 validation records (V1/V2/V3). Cluster bootstrap resamples source videos (uavNNNNNNN / UAVDT Mxxxx)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT/"artifacts/tc2-validation"
RUNS = ROOT/"artifacts/tc2-runs"
ARMS = {"B0": ["B0-SIGMOID"], "REF": ["TC2-P1-REF-001", "TC2-P1-REF-S2-001"],
        "G300": ["TC2-S4-G300-001", "TC2-S4-G300-S2-001"], "GMIX": ["TC2-S4-GMIX-001", "TC2-S4-GMIX-S2-001"]}
STORED = {"B0-SIGMOID": NEW/"B0-SIGMOID-NANSAFE-retry1793333212"}  # stage-2 B0 (sigmoid, NaN-safe) V1+V2
DISAGREE_DV1B, BOOT, BOOT_SEED = 0.015, 2000, 20261104


def stored(label):
    return json.loads(((STORED.get(label) or RUNS/label)/"validation.json").read_text())


def video(key):  # "V1B/uav0000xxx_yyyyy_s", "V2/uav..._v#12", "V3/M0301#19" -> source video
    name = key.split("/", 1)[1].split("#")[0]
    return name[:10] if name.startswith("uav") else name


new = {label: json.loads((NEW/f"V1B-{label}"/"validation.json").read_text()) for arm in ARMS.values() for label in arm}
auc = {label: {k: v["success_auc"] for k, v in rec["per_episode"].items()} for label, rec in new.items()}

# Gate: V1 repeated in this job must reproduce every stored per-episode V1 AUC (same weights, same code path).
gate = {}
for label in new:
    old = {k: v["success_auc"] for k, v in stored(label)["per_episode"].items() if k.startswith("V1/")}
    mine = {k: v for k, v in auc[label].items() if k.startswith("V1/")}
    gate[label] = set(old) == set(mine) and max(abs(old[k]-mine[k]) for k in old) < 1e-9
gate_ok = all(gate.values())
listings = {rec["episode_listing_sha256"] for rec in new.values()}
V1B = sorted(k for k in auc[ARMS["REF"][0]] if k.startswith("V1B/"))
V1 = sorted(k for k in auc[ARMS["REF"][0]] if k.startswith("V1/"))
print(f"gate (V1 reproduces stored per-episode AUC): {gate} -> {'PASS' if gate_ok else 'FAIL (V1B results unreliable)'}")
print(f"listing hashes {len(listings)}; V1B episodes {len(V1B)} from {len({video(k) for k in V1B})} source videos")

# Stored V2/V3 for the M' (SOT-long) reconstruction; B0's stage-2 record has no V3, so B0 gets no M'.
full = {label: dict(auc[label], **{k: v["success_auc"] for k, v in stored(label)["per_episode"].items()
                                   if k.startswith(("V2/", "V3/"))}) for label in new}
group = {f'{e["subset"]}/{e["name"]}': e["group"] for e in stored(ARMS["REF"][0])["episodes"]}
STRATA = {"V1": V1, "V1B": V1B, "SOTlong": V1+V1B,
          "V2veh": sorted(k for k in group if k.startswith("V2/") and group[k] == "vehicle"),
          "V2per": sorted(k for k in group if k.startswith("V2/") and group[k] == "person"),
          "V3": sorted(k for k in group if k.startswith("V3/"))}


def score(label, strata):
    s = {n: float(np.mean([full[label][k] for k in keys])) for n, keys in strata.items() if keys and keys[0] in full[label]}
    if {"SOTlong", "V2veh", "V2per", "V3"} <= set(s):
        s["Mprime"] = float(np.mean([s["SOTlong"], np.mean([s["V2veh"], s["V2per"]]), s["V3"]]))
    return s


def arm_avg(arm, strata=STRATA):
    per = [score(label, strata) for label in ARMS[arm]]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


clusters = {n: sorted({video(k) for k in keys}) for n, keys in STRATA.items()}
rng = np.random.default_rng(BOOT_SEED)


def resample():  # source videos with replacement within each stratum; SOTlong = V1 + V1B draws
    out = {}
    for n in ("V1", "V1B", "V2veh", "V2per", "V3"):
        picked = rng.choice(clusters[n], len(clusters[n]))
        out[n] = [k for v in picked for k in STRATA[n] if video(k) == v]
    out["SOTlong"] = out["V1"]+out["V1B"]
    return out


draws = [resample() for _ in range(BOOT)]
result = {"gate": gate, "gate_ok": gate_ok, "v1b_episodes": len(V1B), "v1b_videos": len(clusters["V1B"]),
          "listing_hashes": len(listings), "per_model": {label: score(label, STRATA) for label in new}, "arms": {}}
ref = arm_avg("REF")
seed_sd = abs(result["per_model"][ARMS["REF"][0]]["V1B"]-result["per_model"][ARMS["REF"][1]]["V1B"])
print(f"\nREF V1B seeds {result['per_model'][ARMS['REF'][0]]['V1B']:.4f} / {result['per_model'][ARMS['REF'][1]]['V1B']:.4f}"
      f" (|diff| {seed_sd:.4f}); V1 {ref['V1']:.4f}; SOT-long {ref['SOTlong']:.4f}")
for arm in ("B0", "G300", "GMIX"):
    a = arm_avg(arm)
    d = {k: a[k]-ref[k] for k in a if k in ref}
    seeds = [score(x, STRATA)["V1B"]-score(r, STRATA)["V1B"] for x, r in zip(ARMS[arm], ARMS["REF"])] if arm != "B0" else []
    boot = {k: [] for k in ("V1B", "SOTlong", "Mprime") if k in d}
    for strata in draws:
        aa, rr = arm_avg(arm, strata), arm_avg("REF", strata)
        for k in boot:
            boot[k].append(aa[k]-rr[k])
    q = {k: [float(np.percentile(v, 10)), float(np.percentile(v, 90)), float(np.std(v))] for k, v in boot.items()}
    rec = {"d": d, "seed_dV1B": seeds, "boot_p10_p90_se": q}
    if arm != "B0":
        rec["disagreement"] = d["V1B"] >= DISAGREE_DV1B and q["V1B"][0] > 0
        rec["v1b_confirms_worse"] = q["V1B"][1] < 0
    result["arms"][arm] = rec
    line = " ".join(f"d{k} {d[k]:+.4f}" for k in ("V1", "V1B", "SOTlong", "Mprime") if k in d)
    print(f"{arm} - REF: {line}; seeds dV1B {['%+.4f' % s for s in seeds]}")
    print("   " + "; ".join(f"{k} P10/P90 {v[0]:+.4f}/{v[1]:+.4f} SE {v[2]:.4f}" for k, v in q.items()))
    if arm != "B0":
        print(f"   V/V1B disagreement (dV1B >= +{DISAGREE_DV1B} and P10 > 0): {rec['disagreement']}; "
              f"V1B confirms arm worse (P90 < 0): {rec['v1b_confirms_worse']}")
print("\nB0 -> REF (fine-tuning gain): V1 %+.4f, V1B %+.4f; sealed test-dev disjoint-34 +0.0749 (stage 2, "
      "reports/SOURCE_VIDEO_AUDIT.md)" % (-result["arms"]["B0"]["d"]["V1"], -result["arms"]["B0"]["d"]["V1B"]))
out = ROOT/"artifacts/stage6"
out.mkdir(parents=True, exist_ok=True)
(out/"stage6_v1b.json").write_text(json.dumps(result, indent=1))
