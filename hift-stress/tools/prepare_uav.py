"""Convert the raw KAUST UAV123 / UAV123@10fps / UAV20L download to the 'flat' layout.

Raw download:  <raw>/data_seq/<SET>/<base>/000001.jpg ...   and   <raw>/anno/<SET>/<seq>.txt
Some annotated sequences are slices of one video (bird1_1, bird1_2, ...). Frame ranges for
UAV123 and UAV20L (30 fps) are bundled in tools/ranges/; UAV123_10fps needs the configSeqs*.m
file from the KAUST tracker-benchmark zip (--config). UAV20L frames live under data_seq/UAV123.

Flat layout written (symlinks, no copying):  <out>/data_seq/<seq>/*.jpg  and  <out>/anno/<seq>.txt

  python tools/prepare_uav.py --raw data/UAV123 --set UAV123 --out data/uav123_flat
      (frame ranges for UAV123 / UAV20L are bundled; for UAV123_10fps pass --config configSeqs.m)
Every sequence is checked: number of frames must equal number of boxes, else it is skipped.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

PAT = re.compile(r"'name'\s*,\s*'([^']+)'\s*,\s*'path'\s*,\s*'([^']*)'\s*,\s*'startFrame'\s*,\s*(\d+)\s*,"
                 r"\s*'endFrame'\s*,\s*(\d+)")


def parse_config(path):
    text = Path(path).read_text(errors="ignore")
    out = {}
    for name, p, start, end in PAT.findall(text):
        base = [s for s in re.split(r"[\\/]", p) if s][-1]
        out[name] = (base, int(start), int(end))
    return out


def parse_ranges_json(path, set_name):
    import json
    data = json.loads(Path(path).read_text())
    key = {"UAV123_10fps": "UAV123"}.get(set_name, set_name)
    if key not in data:
        raise SystemExit(f"{path} has no entry for {key}; keys: {list(data)}")
    if set_name == "UAV123_10fps":
        raise SystemExit("The bundled ranges are for the 30 fps release; use --config for UAV123_10fps.")
    return {n: (v["folder_name"], int(v["start_frame"]), int(v["end_frame"])) for n, v in data[key].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--set", required=True, help="UAV123, UAV123_10fps or UAV20L")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="configSeqs*.m with frame ranges")
    ap.add_argument("--ranges-json", default=str(Path(__file__).parent / "ranges" / "uav123_got10k.json"),
                    help="frame ranges as JSON (default: bundled GOT-10k table, UAV123 and UAV20L at 30 fps)")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    a = ap.parse_args()
    raw = Path(a.raw)
    anno_dir = raw / "anno" / a.set
    seq_dir = raw / "data_seq" / ("UAV123" if a.set == "UAV20L" else a.set)
    if not anno_dir.is_dir() or not seq_dir.is_dir():
        raise SystemExit(f"expected {anno_dir} and {seq_dir}")
    if a.config:
        ranges = parse_config(a.config)
        print(f"{len(ranges)} sequences with frame ranges in {a.config}")
    else:
        ranges = parse_ranges_json(a.ranges_json, a.set)
        print(f"{len(ranges)} sequences with frame ranges in {a.ranges_json}")
    out = Path(a.out)
    (out / "anno").mkdir(parents=True, exist_ok=True)
    done, skipped = 0, []
    for anno in sorted(anno_dir.glob("*.txt")):
        name = anno.stem
        n_boxes = sum(1 for line in anno.read_text().splitlines() if line.strip())
        if name in ranges:
            base, start, end = ranges[name]
        elif (seq_dir / name).is_dir():
            base, start, end = name, 1, None
        else:
            skipped.append(f"{name}: no frame range (need --config)")
            continue
        imgs = sorted(p for p in (seq_dir / base).glob("*.jpg"))
        if end is None:
            end = len(imgs)
        sel = [seq_dir / base / f"{i:06d}.jpg" for i in range(start, end + 1)]
        missing = sum(1 for p in sel if not p.exists())
        if len(sel) != n_boxes or missing:
            skipped.append(f"{name}: frames {base} {start}-{end} ({len(sel)}) vs {n_boxes} boxes, "
                           f"{missing} image files missing")
            continue
        dst = out / "data_seq" / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for k, src in enumerate(sel, 1):
            target = dst / f"{k:06d}.jpg"
            if a.copy:
                shutil.copy2(src, target)
            else:
                os.symlink(src.resolve(), target)
        shutil.copy2(anno, out / "anno" / anno.name)
        done += 1
    print(f"prepared {done} sequences in {out}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
