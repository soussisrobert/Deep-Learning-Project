"""Edge-cost benchmark: size, FLOPs, latency, memory and output fidelity per variant.

  python scripts/bench_edge.py --snapshot weights/general_model.pth --dataset dtb70 --root data/DTB70
  python scripts/bench_edge.py ... --device cpu --threads 1 2 4      # emulate a small onboard CPU
  python scripts/bench_edge.py ... --device cuda --variants fp32 fp16 prune50 compile

Fidelity = how far the variant's score map moves from the fp32 one on real search crops
(1.0 = identical peak location on every crop). It predicts accuracy loss before you
spend time tracking a whole benchmark. Accuracy itself comes from run_track.py.
"""
import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)  # torch quantisation deprecation notices
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiftstress.crops import gt_crops, gt_pairs  # noqa: E402
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.edge import benchmark, count_flops, make_variant, state_size_mb  # noqa: E402
from hiftstress.model import build_model, count_params, pick_device  # noqa: E402
from hiftstress.tracker import score_map  # noqa: E402


@torch.no_grad()
def fidelity(ref, net, dtype, device, z, xs):
    ref.template(z.to(next(ref.parameters()).device))
    net.template(z.to(device, dtype))
    same_peak, rel_err = [], []
    for x in xs:
        a = score_map(ref, x.to(next(ref.parameters()).device)).float().cpu().flatten()
        b = score_map(net, x.to(device, dtype)).float().cpu().flatten()
        same_peak.append(int(a.argmax() == b.argmax()))
        rel_err.append(float((a - b).norm() / (a.norm() + 1e-12)))
    return {"same_peak_fraction": float(np.mean(same_peak)), "score_rel_error": float(np.mean(rel_err))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot")
    p.add_argument("--allow-random", action="store_true")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "dtb70", "flat"])
    p.add_argument("--root")
    p.add_argument("--crop-seqs", type=int, default=20, help="sequences to take real crops from")
    p.add_argument("--device", default="auto")
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--threads", nargs="+", type=int, default=[0], help="CPU thread counts to test")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default="results/edge")
    a = p.parse_args()
    if not a.snapshot and not a.allow_random:
        sys.exit("Give --snapshot, or --allow-random for a pipeline test.")
    device = pick_device(a.device)
    if a.variants is None:
        a.variants = ["fp32", "prune30", "prune50", "prune70"] + (
            ["fp16", "compile"] if device.type == "cuda" else ["fp16"] if device.type == "mps" else ["int8dyn", "int8bb", "int8full"])
    seqs = load_dataset(a.dataset, a.root, a.crop_seqs, None, None, 1)
    zs, xs = gt_crops(seqs, frames_per_seq=3, device="cpu")
    pairs = gt_pairs(seqs, frames_per_seq=2, device="cpu")
    z, x = zs[0], xs[0]
    ref, info = build_model(a.snapshot, device)
    flops = count_flops(ref, z.to(device), x.to(device))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for threads in a.threads:
        if threads:
            torch.set_num_threads(threads)
        for v in a.variants:
            try:
                net, dtype, dev, vinfo = make_variant(v, ref, device, xs[:40], pairs)
            except SystemExit as e:
                print(f"[skip] {v}: {e}")
                continue
            total, nonzero = count_params(net) if any(True for _ in net.parameters()) else (0, 0)
            row = {"variant": v, "device": str(dev), "threads": torch.get_num_threads(),
                   "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                   "params_m": total / 1e6, "nonzero_params_m": nonzero / 1e6,
                   "size_mb": state_size_mb(net), **flops, **vinfo,
                   "weights": info.get("snapshot") or "RANDOM"}
            row.update(benchmark(net, dtype, dev, z, x, iters=a.iters))
            ref_on = ref if dev == device else ref.to(dev)
            row.update(fidelity(ref_on, net, dtype, dev, z, xs[:60]))
            if ref_on is not ref or dev != device:
                ref.to(device)
            rows.append(row)
            tag = f"{dev.type}_t{row['threads']}_{v}"
            (out / f"{tag}.json").write_text(json.dumps(row, indent=2))
            print(f"[{tag}] {row['latency_ms_mean']:7.2f} ms ({row['fps_network']:6.1f} fps) | "
                  f"backbone {row['backbone_ms_mean']:.2f} ms | {row['size_mb']:.1f} MB | "
                  f"same peak {row['same_peak_fraction']:.2f}")
    print(f"wrote {len(rows)} rows to {out}/")


if __name__ == "__main__":
    main()
