"""Re-score saved predictions (results/<run>/preds/*.txt) against the benchmark annotations.

No tracking is re-run. For every run it reports, side by side:
  - our strict rule (target-absent frames count as misses)    -> must reproduce the stored numbers
  - HiFT's official toolkit rule (absent frames count as precision hits) -> comparable with the paper
  - per-attribute success / precision when UAV123 attribute files are given

  python scripts/rescore.py --dataset flat --root data/uav123_flat \
      --att-dir data/raw/UAV123/anno/UAV123/att --results results --out results/rescore

Attributes, preferred source: --uav-json data/raw/UAV123/UAV.json (shipped in the archive; pysot
format, each video has an "attr" list of attribute names). Fallback: --att-dir with 12 0/1 flags per
sequence in this order (check it once against DatasetAnnotation.pdf before quoting):
  SV ARC LR FM FOC POC OV BC IV VC CM SOB
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.metrics import aggregate, sequence_scores  # noqa: E402

ATTRS = ["SV", "ARC", "LR", "FM", "FOC", "POC", "OV", "BC", "IV", "VC", "CM", "SOB"]
ATTR_NAMES = {"SV": "scale variation", "ARC": "aspect ratio change", "LR": "low resolution",
              "FM": "fast motion", "FOC": "full occlusion", "POC": "partial occlusion",
              "OV": "out of view", "BC": "background clutter", "IV": "illumination variation",
              "VC": "viewpoint change", "CM": "camera motion", "SOB": "similar object"}


def read_attrs(att_dir):
    out = {}
    for f in Path(att_dir).glob("*.txt"):
        vals = [int(float(v)) for v in f.read_text().replace(",", " ").split()]
        if len(vals) == len(ATTRS):
            out[f.stem] = dict(zip(ATTRS, vals))
    return out


def read_uav_json(path):
    data = json.loads(Path(path).read_text())
    return {name: {at: 1 for at in v.get("attr", [])} for name, v in data.items() if isinstance(v, dict)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flat")
    ap.add_argument("--root", default="data/uav123_flat")
    ap.add_argument("--results", default="results")
    ap.add_argument("--att-dir")
    ap.add_argument("--uav-json", help="pysot-style UAV.json with per-video 'attr' lists")
    ap.add_argument("--out", default="results/rescore")
    a = ap.parse_args()
    gts = {s.name: s.gt for s in load_dataset(a.dataset, a.root)}
    attrs = read_uav_json(a.uav_json) if a.uav_json else (read_attrs(a.att_dir) if a.att_dir else {})
    attr_keys = sorted({k for v in attrs.values() for k in v}) if a.uav_json else ATTRS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, attr_rows = [], []
    for summ in sorted(glob.glob(f"{a.results}/*/summary.json")):
        run = Path(summ).parent
        stored = json.loads(Path(summ).read_text())
        stored_seq = {p["name"]: p for p in json.loads((run / "per_seq.json").read_text())}
        per = []
        for name in stored_seq:
            pred = np.loadtxt(run / "preds" / f"{name}.txt", delimiter=",", ndmin=2)
            s = sequence_scores(pred, gts[name])
            s["name"] = name
            per.append(s)
        agg = aggregate(per)
        repro = abs(agg["success_auc"] - stored["success_auc"]) < 1e-6 and abs(agg["precision_20"] - stored["precision_20"]) < 1e-6
        row = dict(run=run.name, sequences=agg["sequences"], frames=agg["frames"],
                   success_auc=agg["success_auc"], precision_20_strict=agg["precision_20"],
                   precision_20_official=agg["precision_20_official"],
                   absent_frames=sum(p["absent_frames"] for p in per), reproduces_stored=repro)
        rows.append(row)
        (run / "rescore.json").write_text(json.dumps({**row, "per_seq": [
            {k: p[k] for k in ("name", "frames", "success_auc", "precision_20", "precision_20_official", "absent_frames")}
            for p in per]}, indent=1))
        if attrs:
            for at in attr_keys:
                sel = [p for p in per if attrs.get(p["name"], {}).get(at) == 1]
                if sel:
                    g = aggregate(sel)
                    attr_rows.append(dict(run=run.name, attribute=at, name=ATTR_NAMES.get(at, at), sequences=len(sel),
                                          success_auc=g["success_auc"], precision_20_official=g["precision_20_official"]))
        print(f"{run.name:45s} AUC {agg['success_auc']:.3f}  P20 strict {agg['precision_20']:.3f}  "
              f"official {agg['precision_20_official']:.3f}  reproduces stored: {repro}")
    with open(out / "rescore.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if attr_rows:
        with open(out / "attributes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(attr_rows[0]))
            w.writeheader()
            w.writerows(attr_rows)
    bad = [r["run"] for r in rows if not r["reproduces_stored"]]
    print(f"{len(rows)} runs re-scored -> {out}/" + (f"   WARNING: not reproduced: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
