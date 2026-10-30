"""Collect every results/*/summary.json and results/edge/*.json into tables and figures.

  python scripts/summarize.py --results results
Writes results/tables/{tracking,edge}.csv, results/tables/summary.md and results/figures/*.png
"""
import argparse
import csv
import json
from pathlib import Path

TRACK_COLS = ["run", "dataset", "sequences", "frames", "variant", "attack", "eps", "steps", "defense",
              "adaptive", "success_auc", "precision_20", "fps_end_to_end", "device", "seed"]
EDGE_COLS = ["variant", "device", "gpu", "cpu", "threads", "params_m", "nonzero_params_m", "size_mb",
             "gflops_total", "latency_ms_mean", "latency_ms_p90", "backbone_ms_mean", "head_ms_mean",
             "fps_network", "peak_mem_mb", "same_peak_fraction", "score_rel_error", "weights"]


def load(results):
    runs = [json.loads(p.read_text()) for p in sorted(results.glob("*/summary.json"))]
    edge = [json.loads(p.read_text()) for p in sorted((results / "edge").glob("*.json"))] \
        if (results / "edge").is_dir() else []
    return runs, edge


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c)) for c in cols})


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def figures(runs, edge, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_data = {}
    for r in runs:
        by_data.setdefault(r["dataset"] + (f"-{r['sequences']}seq"), []).append(r)
    for data, rs in by_data.items():
        clean = [r for r in rs if r["attack"] == "none" and r["defense"] == "none" and r["variant"] == "fp32"]
        base = clean[0]["success_auc"] if clean else None
        # 1. robustness curves: success AUC vs eps per attack, no defence, fp32
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for attack in ("noise", "fgsm", "pgd"):
            pts = sorted((r["eps"], r["success_auc"]) for r in rs if r["attack"] == attack
                         and r["defense"] == "none" and r["variant"] == "fp32")
            if pts:
                pts = ([(0, base)] if base is not None else []) + pts
                ax.plot(*zip(*pts), marker="o", label=attack)
        if ax.lines:
            ax.set(xlabel="perturbation budget eps (pixels of 255)", ylabel="success AUC", title=f"HiFT under attack ({data})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"robustness_{data}.png", dpi=200)
        plt.close(fig)
        # 2. defences: clean vs attacked AUC per defence
        defs = sorted({r["defense"] for r in rs if r["variant"] == "fp32"})
        if len(defs) > 1:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            xs = range(len(defs))
            for off, label, pick in ((-0.25, "clean", lambda r: r["attack"] == "none"),
                                     (0.0, "attacked", lambda r: r["attack"] == "pgd" and not r["adaptive"]),
                                     (0.25, "adaptive attack", lambda r: r["attack"] == "pgd" and r["adaptive"])):
                vals = []
                for d in defs:
                    m = [r["success_auc"] for r in rs if r["defense"] == d and r["variant"] == "fp32" and pick(r)]
                    vals.append(max(m) if m else 0)
                ax.bar([x + off for x in xs], vals, width=0.25, label=label)
            ax.set_xticks(list(xs), defs, rotation=20)
            ax.set(ylabel="success AUC", title=f"Input-transform defences ({data})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"defences_{data}.png", dpi=200)
            plt.close(fig)
    # 3. accuracy vs latency per variant (needs both tracking and edge rows)
    acc = {}
    for r in runs:
        if r["attack"] == "none" and r["defense"] == "none":
            acc.setdefault(r["variant"], r["success_auc"])
    pts = [(e["latency_ms_mean"], acc[e["variant"]], f"{e['variant']} ({e['device']}, t{e['threads']})")
           for e in edge if e["variant"] in acc]
    if pts:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for lat, auc, label in pts:
            ax.scatter(lat, auc)
            ax.annotate(label, (lat, auc), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="network latency per frame (ms)", ylabel="success AUC (clean)", title="Accuracy vs edge cost")
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_vs_latency.png", dpi=200)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()
    results = Path(a.results)
    runs, edge = load(results)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    runs.sort(key=lambda r: (r["dataset"], r["variant"], r["attack"], r["eps"], r["defense"], r["adaptive"]))
    write_csv(tables / "tracking.csv", runs, TRACK_COLS)
    write_csv(tables / "edge.csv", edge, EDGE_COLS)
    short_t = ["run", "variant", "attack", "eps", "defense", "adaptive", "success_auc", "precision_20", "fps_end_to_end"]
    short_e = ["variant", "device", "threads", "size_mb", "gflops_total", "latency_ms_mean", "fps_network",
               "same_peak_fraction"]
    (tables / "summary.md").write_text("## Tracking\n\n" + md_table(runs, short_t) +
                                       "\n\n## Edge cost\n\n" + md_table(edge, short_e) + "\n")
    figures(runs, edge, results / "figures")
    random_runs = [r["run"] for r in runs if (r.get("model") or {}).get("warning")]
    if random_runs or any(e.get("weights") == "RANDOM" for e in edge):
        print("WARNING: some results use RANDOM weights (pipeline tests). Do not report them.")
    print(f"{len(runs)} tracking runs, {len(edge)} edge rows -> {tables}/ and {results / 'figures'}/")


if __name__ == "__main__":
    main()
