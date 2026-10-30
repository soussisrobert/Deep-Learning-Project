"""Benchmark loaders. Each returns a list of Sequence(name, frames, gt).

Supported layouts
- synthetic : generated on the fly, no download (pipeline tests)
- dtb70     : <root>/<seq>/img/*.jpg + <root>/<seq>/groundtruth_rect.txt
- flat      : <root>/data_seq/<seq>/*.jpg + <root>/anno/<seq>.txt
              (the layout HiFT's own UAV123@10fps / UAV20L loaders expect;
               tools/prepare_uav.py converts the raw KAUST download to it)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sequence:
    name: str
    frames: list            # list of paths, or list of np.ndarray (synthetic)
    gt: np.ndarray          # (N, 4) x, y, w, h ; NaN/0 rows = target absent
    attrs: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.frames)

    def image(self, i):
        f = self.frames[i]
        if isinstance(f, np.ndarray):
            return f
        img = cv2.imread(str(f))
        if img is None:
            raise IOError(f"could not read {f}")
        return img


def read_boxes(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [v for v in re.split(r"[,\s]+", line) if v]
        rows.append([float("nan") if v.lower() == "nan" else float(v) for v in vals[:4]])
    return np.array(rows, dtype=np.float64)


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_dtb70(root):
    seqs = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        gt_file = d / "groundtruth_rect.txt"
        if not gt_file.exists():
            continue
        frames, gt = _images(d / "img"), read_boxes(gt_file)
        n = min(len(frames), len(gt))
        if len(frames) != len(gt):
            print(f"[dtb70] {d.name}: {len(frames)} frames vs {len(gt)} boxes, using first {n}")
        seqs.append(Sequence(d.name, frames[:n], gt[:n]))
    return seqs


def load_flat(root):
    root = Path(root)
    seqs = []
    for anno in sorted((root / "anno").glob("*.txt")):
        folder = root / "data_seq" / anno.stem
        if not folder.is_dir():
            print(f"[flat] skip {anno.stem}: no frames folder")
            continue
        frames, gt = _images(folder), read_boxes(anno)
        if len(frames) != len(gt):
            raise SystemExit(f"[flat] {anno.stem}: {len(frames)} frames vs {len(gt)} boxes. "
                             "Re-run tools/prepare_uav.py.")
        seqs.append(Sequence(anno.stem, frames, gt))
    return seqs


def _texture(rng, h, w, scale):
    small = rng.integers(0, 256, size=(max(2, h // scale), max(2, w // scale), 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def make_synthetic(n_seqs=3, n_frames=40, size=(360, 480), seed=0):
    """A textured box drifting over a textured background, with exact boxes."""
    rng = np.random.default_rng(seed)
    H, W = size
    seqs = []
    for s in range(n_seqs):
        bg = _texture(rng, H, W, 24)
        tw, th = int(rng.integers(30, 60)), int(rng.integers(30, 60))
        target = _texture(rng, th, tw, 6)
        cv2.rectangle(target, (0, 0), (tw - 1, th - 1), (255, 255, 255), 2)
        x0, y0 = rng.uniform(60, W - 60 - tw), rng.uniform(60, H - 60 - th)
        ax, ay = rng.uniform(20, 60), rng.uniform(15, 40)
        frames, gt = [], []
        for t in range(n_frames):
            x = int(np.clip(x0 + ax * np.sin(t / 7.0), 0, W - tw))
            y = int(np.clip(y0 + ay * np.cos(t / 9.0), 0, H - th))
            img = bg.copy()
            img[y:y + th, x:x + tw] = target
            frames.append(img)
            gt.append([x, y, tw, th])
        seqs.append(Sequence(f"synthetic_{s}", frames, np.array(gt, dtype=np.float64)))
    return seqs


def load_dataset(kind, root=None, max_seqs=None, seqs=None, max_frames=None, seed=0):
    if kind == "synthetic":
        out = make_synthetic(n_seqs=max_seqs or 3, seed=seed)
    elif kind == "dtb70":
        out = load_dtb70(root)
    elif kind == "flat":
        out = load_flat(root)
    else:
        raise SystemExit(f"unknown dataset kind {kind}")
    if not out:
        raise SystemExit(f"no sequences found for {kind} at {root}")
    if seqs:
        wanted = set(seqs)
        out = [s for s in out if s.name in wanted]
        missing = wanted - {s.name for s in out}
        if missing:
            raise SystemExit(f"sequences not found: {sorted(missing)}")
    elif max_seqs:
        # spread the subset across the alphabetic list instead of taking the first N
        idx = np.linspace(0, len(out) - 1, num=min(max_seqs, len(out))).round().astype(int)
        out = [out[i] for i in sorted(set(idx))]
    if max_frames:
        out = [Sequence(s.name, s.frames[:max_frames], s.gt[:max_frames], s.attrs) for s in out]
    return out
