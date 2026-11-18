"""Write the official UAV123 per-video attribute labels to results/attributes_per_video.json.

  python scripts/export_attributes.py            # finds UAV.json (pysot) or anno/UAV123/att under data/

Used to test, with real labels instead of proxies, whether accuracy losses concentrate on
small / occluded / fast targets (round-4 follow-up). No tracking is run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rescore import read_attrs, read_uav_json  # noqa: E402


def main():
    data = ROOT / "data"
    js = sorted(data.rglob("UAV.json"))
    att = sorted(p for p in data.rglob("att") if p.is_dir() and p.parent.name == "UAV123")
    if js:
        attrs, src = read_uav_json(js[0]), str(js[0].relative_to(ROOT))
    elif att:
        attrs, src = read_attrs(att[0]), str(att[0].relative_to(ROOT))
    else:
        raise SystemExit("no UAV123 attribute labels found under data/ (UAV.json or anno/UAV123/att)")
    out = ROOT / "results" / "attributes_per_video.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(source=src, videos=attrs), indent=1))
    names = sorted({a for v in attrs.values() for a, f in v.items() if f})
    print(f"{len(attrs)} videos with labels from {src}; attributes: {', '.join(names)} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
