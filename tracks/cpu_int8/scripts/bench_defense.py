"""Time each input defence per frame (one 1x3x287x287 search crop) next to the network it protects.

  python scripts/bench_defense.py --device cuda --defenses jpeg75 tjpeg75 tjpeg90 blur1 --out results/defense_bench
  python scripts/bench_defense.py --device cpu --threads 4 --variant int8bb ...   (needs --dataset/--root for calibration)

For each defence it reports the defence time and its share of one tracking step of the model on the
same device (the overhead rule in ROUND3_PLAN.md). The crop is a real UAV123 search crop when a
dataset is given, else random pixels. Timing: warm-up, then the median over --iters runs.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiftstress.defenses import make_defense  # noqa: E402
from hiftstress.edge import make_variant  # noqa: E402
from hiftstress.model import build_model, pick_device  # noqa: E402


def sync(d):
    if d.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, d, iters, warm=10):
    for _ in range(warm):
        fn()
    sync(d)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(d)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.mean(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="weights/general_model.pth")
    ap.add_argument("--allow-random", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--dataset", default="flat")
    ap.add_argument("--root")
    ap.add_argument("--defenses", nargs="+", default=["jpeg75", "jpeg85", "jpeg90", "jpeg95", "tjpeg75", "tjpeg85",
                                                         "tjpeg90", "tjpeg95", "blur1", "median3", "bits5"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="results/defense_bench")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    net, _ = build_model(None if a.allow_random else a.snapshot, dev)
    z, x, calib, pairs = torch.rand(1, 3, 127, 127) * 255, torch.rand(1, 3, 287, 287) * 255, None, None
    if a.root:
        from hiftstress.crops import gt_crops, gt_pairs
        from hiftstress.data import load_dataset
        seqs = load_dataset(a.dataset, a.root, 8, None, None, 0)
        zs, xs = gt_crops(seqs, 2, "cpu")
        z, x = zs[0], xs[0]
        calib, pairs = xs, gt_pairs(seqs, 2, "cpu")
    net, dtype, rdev, _ = make_variant(a.variant, net, dev, calib, pairs)
    z, x = z.to(rdev, dtype), x.to(rdev, dtype)
    with torch.no_grad():
        net.template(z)
        model_ms, _ = timeit(lambda: net.track(x), rdev, a.iters)
    rows = []
    for spec in a.defenses:
        d = make_defense(spec)
        with torch.no_grad():
            med, mean = timeit(lambda: d(x), rdev, a.iters)
            err = float((d(x) - x).abs().mean())
        rows.append(dict(defense=d.name, device=str(rdev), threads=torch.get_num_threads(), variant=a.variant,
                         defense_ms=med, defense_ms_mean=mean, model_ms=model_ms, share_of_model=med / model_ms,
                         mean_abs_change=err))
        print(f"{d.name:9s} {str(rdev):6s} t{torch.get_num_threads()} {a.variant:7s} defence {med:7.3f} ms | "
              f"network {model_ms:7.2f} ms | share {med / model_ms:6.1%} | mean |change| {err:.2f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{rdev.type}_t{torch.get_num_threads()}_{a.variant}"
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps({"bench": tag, "rows": rows}, indent=1))  # marks the queue item done
    print(f"wrote {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
