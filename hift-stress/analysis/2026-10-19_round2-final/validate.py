"""Independent validation of the final round-2 package (no project code imported)."""
import json, sys, numpy as np
from pathlib import Path
R = Path(sys.argv[1])
S40 = json.loads((R.parent / "source/attack-subset-40.json").read_text())["sequences"]
runs = sorted(p.name for p in R.iterdir() if (p / "summary.json").exists())
L = lambda n: (json.loads((R/n/"summary.json").read_text()), {p["name"]: p for p in json.loads((R/n/"per_seq.json").read_text())})
problems = []
print(f"{len(runs)} runs")
rows = []
for n in runs:
    s, per = L(n)
    exp = 40 if n.startswith("uav123-sub40") else 123
    if s["sequences"] != exp or len(per) != exp: problems.append(f"{n}: {len(per)} seqs")
    if exp == 40 and sorted(per) != sorted(S40): problems.append(f"{n}: not the fixed 40")
    fr = sum(p["frames"] for p in per.values())
    if fr != s["frames"] or fr != (37681 if exp == 40 else 112578): problems.append(f"{n}: frames {fr}")
    npred = len(list((R/n/"preds").glob("*.txt")))
    if npred != exp: problems.append(f"{n}: {npred} pred files")
    m = np.mean([p["success_auc"] for p in per.values()])
    if abs(m - s["success_auc"]) > 1e-9: problems.append(f"{n}: mean of per-seq {m} != {s['success_auc']}")
    # name <-> settings consistency
    parts = n.split("_")
    rows.append(dict(run=n, variant=s["variant"], attack=s["attack"], eps=s["eps"], steps=s["steps"], adaptive=s["adaptive"],
                     defense=s["defense"], device=s["device"], attack_device=s.get("attack_device"), seed=s["seed"],
                     auc=s["success_auc"], p20=s["precision_20"], p20o=s.get("precision_20_official"),
                     ms=s["ms_per_frame"], calib=s.get("variant_info", {}).get("calibration_sequences")))
    if ("adaptive" in n) != bool(s["adaptive"]): problems.append(f"{n}: adaptive flag {s['adaptive']}")
    for d in ("blur1", "median3", "bits5", "jpeg75"):
        if d in n and d.rstrip("0123456789") not in str(s["defense"]): problems.append(f"{n}: defense {s['defense']}")
    if "pgd" in n and (s["attack"] != "pgd" or s["steps"] != 10): problems.append(f"{n}: attack {s['attack']} steps {s['steps']}")
    if "pgd4" in n and abs(float(s["eps"]) - 4) > 1e-9: problems.append(f"{n}: eps {s['eps']}")
    if "_t4" in n and s["device"] != "cpu": problems.append(f"{n}: device {s['device']}")
    if "_t4" not in n and s["device"] != "cuda": problems.append(f"{n}: device {s['device']}")
    if "seed1" in n and s["seed"] != 1 or "seed2" in n and s["seed"] != 2: problems.append(f"{n}: seed {s['seed']}")
json.dump(rows, open(sys.argv[2], "w"), indent=1, default=str)
for r in rows:
    print(f"{r['run']:46s} {r['variant']:8s} {r['attack']:5s} eps={r['eps']!s:4s} st={r['steps']!s:3s} ad={int(r['adaptive'])} def={r['defense']!s:12s} {r['device']:4s} atk={r['attack_device']!s:5s} seed={r['seed']} AUC={r['auc']:.4f} P20o={r['p20o'] if r['p20o'] is None else round(r['p20o'],4)}  def_ms={r['ms']['defense']:.2f} tot_ms={r['ms']['total']:.2f}")
print("PROBLEMS:", problems or "none")
# calibration overlap
for r in rows:
    if r["calib"]:
        ov = set(r["calib"]) & set(S40)
        print(f"calib {r['run']}: {len(r['calib'])} seqs, overlap with fixed40: {len(ov)}")
