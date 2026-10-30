"""Pipeline checks that need no dataset and no weights (about 1 minute on a laptop CPU).

  python scripts/smoke_test.py                    # random weights
  python scripts/smoke_test.py --snapshot weights/general_model.pth   # also checks the real weights load

Every check prints PASS or FAIL; the exit code is non-zero if anything fails.
"""
import argparse
import sys
import warnings
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)  # torch quantisation deprecation notices
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiftstress.attacks import PGD, RandomSignNoise  # noqa: E402
from hiftstress.crops import gt_crops  # noqa: E402
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.defenses import make_defense  # noqa: E402
from hiftstress.edge import make_variant  # noqa: E402
from hiftstress.metrics import sequence_scores  # noqa: E402
from hiftstress.model import HIFT_ROOT, build_model, pick_device  # noqa: E402
from hiftstress.tracker import TrackingModel, run_sequence, score_map  # noqa: E402

RESULTS = []


def check(name):
    def wrap(fn):
        try:
            msg = fn() or ""
            RESULTS.append(True)
            print(f"PASS  {name} {msg}")
        except Exception as e:  # noqa: BLE001
            RESULTS.append(False)
            print(f"FAIL  {name}: {e}")
            traceback.print_exc(limit=2)
        return fn
    return wrap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = pick_device(a.device)
    print(f"device: {device} | torch {torch.__version__} | HiFT at {HIFT_ROOT}")

    @check("HiFT patched for any device")
    def _():
        src = (HIFT_ROOT / "pysot/models/utile/utile.py").read_text()
        assert ".cuda()" not in src, "run tools/patch_hift.py"

    @check("metrics: perfect prediction")
    def _():
        gt = np.array([[10, 10, 50, 40]] * 30, float)
        s = sequence_scores(gt, gt)
        assert abs(s["success_auc"] - 20 / 21) < 1e-9 and s["precision_20"] == 1.0, s
        return f"(AUC {s['success_auc']:.3f}, the pysot maximum)"

    @check("metrics: missing ground truth counts as failure")
    def _():
        gt = np.array([[10, 10, 50, 40]] * 10 + [[np.nan] * 4] * 10, float)
        s = sequence_scores(np.nan_to_num(gt), gt)
        assert abs(s["precision_20"] - 0.5) < 1e-9, s

    net, info = build_model(a.snapshot, device)
    seqs = load_dataset("synthetic", max_seqs=2)
    zs, xs = gt_crops(seqs, frames_per_seq=4, device=device)

    @check("weights loaded" if a.snapshot else "random-weight model builds")
    def _():
        if a.snapshot:
            assert not info.get("missing"), info
        net.template(zs[0])
        s = score_map(net, xs[0])
        assert s.shape[-2:] == (11, 11), s.shape
        return f"(score map {tuple(s.shape)})"

    @check("noise control stays inside the budget")
    def _():
        x = xs[0]
        y = RandomSignNoise(4)(x)
        assert float((y - x).abs().max()) <= 4 + 1e-4 and y.min() >= 0 and y.max() <= 255

    @check("PGD stays inside the budget and lowers the target score")
    def _():
        net.template(zs[0])
        atk = PGD(4, steps=5)
        lowered = 0
        for x in xs:
            y = atk(x, lambda inp: score_map(net, inp))
            assert float((y - x).abs().max()) <= 4 + 1e-3 and y.min() >= 0 and y.max() <= 255
        for start, final in atk.last_drop:
            lowered += final < start
        assert lowered >= 0.9 * len(atk.last_drop), atk.last_drop
        return f"({lowered}/{len(atk.last_drop)} crops lowered)"

    @check("defences keep shape and pixel range")
    def _():
        for spec in ("blur1", "median3", "bits5", "jpeg75"):
            d = make_defense(spec)
            y = d(xs[0])
            assert y.shape == xs[0].shape and y.min() >= 0 and y.max() <= 255.5, spec
            g = d.straight_through(xs[0].clone().requires_grad_(True))
            g.sum().backward()

    @check("adaptive PGD through a defence")
    def _():
        net.template(zs[0])
        d = make_defense("blur1")
        y = PGD(4, steps=2, adaptive=True)(xs[0], lambda inp: score_map(net, inp), d)
        assert y.shape == xs[0].shape

    @check("round 5: per-frame confidence, confidence-preserving attack, night corruption")
    def _():
        from hiftstress.corruptions import make_corruption
        from hiftstress.tracker import TrackingModel, confidence_stats
        net.template(zs[0])
        out = net.track(xs[0])
        pk, psr = confidence_stats(out)
        assert np.isfinite(pk) and np.isfinite(psr)
        y = PGD(4, steps=2, objective="shift", shift_lambda=3)(xs[0], lambda inp: score_map(net, inp))
        assert float((y - xs[0]).abs().max()) <= 4 + 1e-3
        z = make_corruption("night:3")(xs[0])
        assert z.shape == xs[0].shape and z.min() >= 0 and z.max() <= 255
        m = TrackingModel(net, device)
        m.template(zs[0]); m.track(xs[0])
        assert len(m.frame_conf) == 1
        return f"(peak {pk:.3f}, PSR {psr:.1f})"

    @check("round 3: torch JPEG matches OpenCV; surrogate gradient + EOT attack; corruptions")
    def _():
        from hiftstress.corruptions import make_corruption
        x = xs[0]
        a, b = make_defense("jpeg75")(x), make_defense("tjpeg75")(x)
        psnr = 10 * float(torch.log10(255 ** 2 / ((a - b) ** 2).mean().clamp_min(1e-9)))
        assert psnr > 38 and b.device == x.device, psnr
        net.template(zs[0])
        y = PGD(4, steps=2, adaptive=True, grad="diff", eot=2, eot_jitter=5)(x, lambda inp: score_map(net, inp),
                                                                             make_defense("jpeg75"))
        assert float((y - x).abs().max()) <= 4 + 1e-3
        for c in ("gauss_noise:3", "gauss_blur:3", "motion_blur:3", "low_light:3", "contrast:3"):
            z = make_corruption(c)(x)
            assert z.shape == x.shape and z.min() >= 0 and z.max() <= 255, c
        return f"(torch vs OpenCV JPEG {psnr:.1f} dB)"

    variants = ["fp32", "prune50"] + (["fp16"] if device.type in ("cuda", "mps") else []) + ["int8dyn", "int8bb", "int8bb-s1", "int8full"]
    for v in variants:
        @check(f"variant {v} runs")
        def _(v=v):
            m, dtype, dev, vinfo = make_variant(v, net, device, [x.cpu() for x in xs],
                                               calib_pairs=[(zs[0].cpu(), x.cpu()) for x in xs])
            m.template(zs[0].to(dev, dtype))
            out = m.track(xs[0].to(dev, dtype))
            assert out["cls1"].shape[-2:] == (11, 11)
            net.to(device)

    @check("end-to-end tracking with attack + defence")
    def _():
        model = TrackingModel(net, device, torch.float32, PGD(2, steps=2), make_defense("bits5"))
        preds, scores, times = run_sequence(seqs[0], model, device)
        assert preds.shape == (len(seqs[0]), 4)
        return f"({len(times)} frames, {1 / np.mean(times[1:]):.1f} fps incl. attack)"

    ok = all(RESULTS)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed" + ("" if ok else "  <-- fix the FAILs before running jobs"))
    if not a.snapshot:
        print("Note: random weights. Re-run with --snapshot once general_model.pth is downloaded.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
