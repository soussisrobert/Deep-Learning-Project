"""Stage-7 close-out (reports/STAGE7_CLOSEOUT.md): sealed-test confidence intervals for frozen REF vs B0.
No inference, no training. Re-reads the stage-2 sealed predictions in artifacts/b2-eval, gates them on identity and
exact re-scoring, then reports a paired source-video cluster bootstrap of REF - B0 (sequence-macro metrics)."""
import json
import re
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from hift_research.evaluation import aggregate_sequences, evaluate_sequence  # noqa: E402
from hift_research.evaluation.assets import read_boxes, sha256_file  # noqa: E402

EVAL = ROOT/"artifacts/b2-eval"
MODELS = {"B0": "B0", "REF1": "TC2-P1-REF-001", "REF2": "TC2-P1-REF-S2-001"}
DATASETS = {"VISDRONE": "VisDrone2019-SOT-test-dev", "UAV123": "UAV123@10fps"}
OVERLAP = "uav0000079_02568_s"  # source video in REF training (reports/SOURCE_VIDEO_AUDIT.md)
METRICS = ("success_auc", "precision_at_20px")
CONTRASTS = {"REF-B0": ("REF", "B0"), "REF1-B0": ("REF1", "B0"), "REF2-B0": ("REF2", "B0"), "REF1-REF2": ("REF1", "REF2")}
BOOT, BOOT_SEED, TOL = 10000, 20261105, 1e-12


def run_name(dataset, model):
    return f"STAGE2-SEALED-{dataset}-{MODELS[model]}-sigmoid"


def cluster(dataset, name):
    """VisDrone: source flight video uavNNNNNNN. UAV123: name without a trailing _<k> segment (car1_2 -> car1)."""
    if dataset == "VISDRONE":
        if not re.fullmatch(r"uav\d{7}_\d{5}_[sv]", name):
            raise ValueError(f"not a VisDrone sequence name: {name!r}")
        return name[:10]
    if not re.fullmatch(r"[a-z]+\d+(_\d+)?", name):
        raise ValueError(f"not a UAV123 sequence name: {name!r}")
    return re.sub(r"_\d+$", "", name)


def local(recorded, root=ROOT):
    """Map a recorded F:\\Code\\HiFT\\... path onto this checkout; refuse anything outside the repo."""
    parts = PureWindowsPath(recorded).parts
    tail = [i for i in range(len(parts)-1) if parts[i] == "Code" and parts[i+1] == "HiFT"]
    if not tail or ".." in parts:
        raise ValueError(f"recorded path is not inside Code\\HiFT: {recorded!r}")
    return Path(root, *parts[tail[-1]+2:])


def check_run(report, run_dir, ground_truth, root=ROOT):
    """Gate G1-G4 for one stored sealed evaluation. Returns a list of failures (empty = pass)."""
    fails = []
    if report.get("prediction_coordinate_origin") != "annotation_native" or report.get("invalid_policy") != "count_as_failure":
        fails.append("G4 unexpected coordinate origin or invalid policy")
    manifest = run_dir/"manifest.json"
    if sha256_file(manifest) != report["inference_run_manifest"]["sha256"]:
        fails.append("G1 inference manifest hash")
    checkpoint = json.loads(manifest.read_text())["checkpoint"]
    if sha256_file(local(checkpoint["path"], root)) != checkpoint["sha256"]:
        fails.append("G3 checkpoint hash")
    if set(report["prediction_inputs"]) != set(report["sequences"]) or set(report["sequences"]) != set(ground_truth):
        fails.append("G4 sequence sets differ")
        return fails
    scores = {}
    for name, rec in sorted(report["prediction_inputs"].items()):
        csv = run_dir/f"{name}.csv"
        if sha256_file(csv) != rec["sha256"]:
            fails.append(f"G1 prediction hash {name}")
            continue
        scores[name] = evaluate_sequence(np.asarray(ground_truth[name], dtype=float), read_boxes(csv),
                                         invalid_policy="count_as_failure")
        for m in METRICS:
            if not abs(scores[name][m]-report["sequences"][name][m]) < TOL:
                fails.append(f"G4 {name} {m}")
    if len(scores) == len(ground_truth):
        agg = aggregate_sequences(scores)
        for m in METRICS:
            if not abs(agg[m]-report["aggregate"][m]) < TOL:
                fails.append(f"G4 aggregate {m}")
    return fails


def cluster_index(clusters):
    """Sorted cluster labels and each sequence's cluster index."""
    labels = sorted(set(clusters))
    return labels, np.array([labels.index(c) for c in clusters], dtype=int)


def draw_weights(n_clusters, draws, rng):
    """Cluster multiplicities for each draw: n_clusters picks with replacement. Shared by every contrast and metric
    of a test set, so all deltas of one draw see the same resampled sequences (paired)."""
    picks = rng.integers(0, n_clusters, size=(draws, n_clusters))
    return np.stack([np.bincount(p, minlength=n_clusters) for p in picks]).astype(float)


def boot_means(deltas, idx, weights):
    """Sequence-macro mean of each draw: every sequence of a picked cluster enters with the cluster's multiplicity."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or deltas.size == 0 or idx.shape != deltas.shape or not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must be a non-empty finite vector aligned with the cluster index")
    n = weights.shape[1]
    return (weights@np.bincount(idx, deltas, n))/(weights@np.bincount(idx, minlength=n).astype(float))


def leave_one_out(deltas, clusters):
    deltas, clusters = np.asarray(deltas, dtype=float), np.asarray(clusters)
    labels = sorted(set(clusters.tolist()))
    if len(labels) < 2:
        raise ValueError("leave-one-cluster-out needs at least two clusters")
    vals = [float(deltas[clusters != c].mean()) for c in labels]
    return min(vals), max(vals)


def verdict(lower95, upper95, seed_deltas):
    """Preregistered reading: improves needs lower95 > 0 and every seed delta > 0; worse needs upper95 < 0."""
    if lower95 > 0 and all(d > 0 for d in seed_deltas):
        return "REF improves over B0"
    if upper95 < 0:
        return "REF worse than B0"
    return "no difference detectable at this sample size"


def summarize(values, point):
    p = np.percentile(values, [2.5, 10, 90, 97.5])
    return {"delta": point, "p2_5": float(p[0]), "p10": float(p[1]), "p90": float(p[2]), "p97_5": float(p[3]),
            "se": float(np.std(values, ddof=1))}


def main():
    target = ROOT/"artifacts/stage7/closeout.json"
    if target.exists():
        raise SystemExit(f"{target} exists; runs are never overwritten")
    sets = {"VISDRONE-34": "VISDRONE", "VISDRONE-35": "VISDRONE", "UAV123": "UAV123"}
    reports, gate = {}, {}
    for ds in DATASETS:
        reports[ds] = {m: json.loads((EVAL/f"{run_name(ds, m)}.evaluation.json").read_text()) for m in MODELS}
        recorded = {r["dataset_manifest"]["sha256"] for r in reports[ds].values()}
        path = local(reports[ds]["B0"]["dataset_manifest"]["path"])
        fails = [] if recorded == {sha256_file(path)} else ["G2 dataset manifest hash"]
        manifest = json.loads(path.read_text())
        if manifest["dataset"] != DATASETS[ds]:
            fails.append("G2 dataset name")
        gt = {row["name"]: row["ground_truth"] for row in manifest["sequences"]}
        for m in MODELS:
            gate[f"{ds}/{m}"] = fails+check_run(reports[ds][m], EVAL/run_name(ds, m), gt)
    gate_ok = not any(gate.values())
    print("gate:", "PASS" if gate_ok else "FAIL", {k: v for k, v in gate.items() if v})
    result = {"gate": gate, "gate_ok": gate_ok, "boot": BOOT, "boot_seed": BOOT_SEED, "sets": {}}
    if gate_ok:
        rng = np.random.default_rng(BOOT_SEED)  # one stream, consumed in the fixed set order below
        for label, ds in sets.items():
            names = sorted(n for n in reports[ds]["B0"]["sequences"] if not (label == "VISDRONE-34" and n == OVERLAP))
            clusters = [cluster(ds, n) for n in names]
            per = {m: {k: np.array([reports[ds][m]["sequences"][n][k] for n in names]) for k in METRICS} for m in MODELS}
            per["REF"] = {k: (per["REF1"][k]+per["REF2"][k])/2 for k in METRICS}
            labels, idx = cluster_index(clusters)
            weights = draw_weights(len(labels), BOOT, rng)
            out = {"sequences": len(names), "clusters": len(labels),
                   "macro": {m: {k: float(v.mean()) for k, v in per[m].items()} for m in per}, "contrasts": {}}
            for cname, (a, b) in CONTRASTS.items():
                out["contrasts"][cname] = {}
                for k in METRICS:
                    d = per[a][k]-per[b][k]
                    out["contrasts"][cname][k] = dict(summarize(boot_means(d, idx, weights), float(d.mean())),
                                                      loco_min_max=leave_one_out(d, clusters))
            for k in METRICS:
                c = out["contrasts"]["REF-B0"][k]
                seeds = [out["contrasts"]["REF1-B0"][k]["delta"], out["contrasts"]["REF2-B0"][k]["delta"]]
                c["verdict"] = verdict(c["p2_5"], c["p97_5"], seeds)
            result["sets"][label] = out
            mac = out["macro"]
            print(f"\n{label}: {len(names)} sequences, {len(labels)} clusters")
            for k in METRICS:
                print(f"  {k}: B0 {mac['B0'][k]:.4f}  REF s1/s2 {mac['REF1'][k]:.4f}/{mac['REF2'][k]:.4f}  mean {mac['REF'][k]:.4f}")
                for cname in CONTRASTS:
                    c = out["contrasts"][cname][k]
                    print(f"    {cname:9s} {c['delta']:+.4f}  95% [{c['p2_5']:+.4f}, {c['p97_5']:+.4f}]  P10/P90 "
                          f"{c['p10']:+.4f}/{c['p90']:+.4f}  SE {c['se']:.4f}  LOCO {c['loco_min_max'][0]:+.4f}/"
                          f"{c['loco_min_max'][1]:+.4f}" + (f"  -> {c['verdict']}" if "verdict" in c else ""))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
