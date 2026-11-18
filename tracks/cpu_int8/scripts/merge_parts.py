"""Merge a condition that was split by sequences into one result folder.

  python scripts/merge_parts.py --out results/<run> results/<run>__p1 results/<run>__p2 ...

Checks the parts share the same condition and have no sequence in common, then writes
<out>/summary.json, per_seq.json and preds/ exactly as a single run would.
Timing per frame is the frame-weighted mean of the parts.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiftstress.metrics import aggregate  # noqa: E402

SAME = ["variant", "attack", "eps", "steps", "adaptive", "defense", "seed", "dataset"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-seqs-file", help="JSON/txt list the merged run must cover exactly")
    a = ap.parse_args()
    parts = [Path(p) for p in a.parts]
    sums = [json.loads((p / "summary.json").read_text()) for p in parts]
    for k in SAME:
        vals = {json.dumps(s.get(k)) for s in sums}
        if len(vals) > 1:
            sys.exit(f"parts disagree on {k}: {vals}")
    per_seq, seen = [], set()
    for p in parts:
        for s in json.loads((p / "per_seq.json").read_text()):
            if s["name"] in seen:
                sys.exit(f"sequence {s['name']} appears in two parts")
            seen.add(s["name"])
            per_seq.append(s)
    if a.expect_seqs_file:
        txt = Path(a.expect_seqs_file).read_text()
        want = set(json.loads(txt)["sequences"] if a.expect_seqs_file.endswith(".json") else txt.split())
        if want != seen:
            sys.exit(f"merged set differs from {a.expect_seqs_file}: missing {sorted(want - seen)[:10]}, "
                     f"extra {sorted(seen - want)[:10]}")
    out = Path(a.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    for p in parts:
        for f in (p / "preds").glob("*.txt"):
            shutil.copy2(f, out / "preds" / f.name)
    summary = dict(sums[0])
    summary.update(aggregate(per_seq))
    n = [s["frames"] - s["sequences"] for s in sums]   # tracked frames (first frame is the init)
    tot = max(sum(n), 1)
    summary["ms_per_frame"] = {k: sum(s["ms_per_frame"][k] * w for s, w in zip(sums, n)) / tot
                               for k in sums[0]["ms_per_frame"]}
    summary["fps_end_to_end"] = 1000.0 / summary["ms_per_frame"]["total"]
    summary["total_minutes"] = sum(s["total_minutes"] for s in sums)
    summary["run"] = out.name
    summary["merged_from"] = [p.name for p in parts]
    if all("attack_objective" in s for s in sums):
        w = [s["frames"] for s in sums]
        summary["attack_objective"] = {k: sum(s["attack_objective"][k] * x for s, x in zip(sums, w)) / sum(w)
                                       for k in sums[0]["attack_objective"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_seq.json").write_text(json.dumps(per_seq, indent=1))
    print(f"merged {len(parts)} parts -> {out}: {len(per_seq)} sequences, AUC {summary['success_auc']:.3f}")


if __name__ == "__main__":
    main()
