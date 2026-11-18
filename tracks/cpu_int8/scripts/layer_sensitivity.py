"""Per-layer int8 sensitivity of HiFT's AlexNet backbone (see LAYER_SENSITIVITY_PLAN.md).

Configurations (which backbone layers stay fp32; everything else in the backbone is static int8, head fp32):
  none   : all five layers int8                (= int8bb)
  keep<k>: layer k stays fp32, others int8      (keep1 = int8bb-s1, our selected recipe)
  only<k>: only layer k is int8, others fp32    (isolates the error layer k introduces)

Calibration is exactly the round-4 DEV pilot recipe (run_track.py code path): 40 videos x 4 spread frames, draws 1-3,
pool = all UAV123 videos minus the DEV evaluation videos minus the TEST videos (--calib-exclude).

  --stage fidelity : all 11 configs x 3 draws. Score-map fidelity vs fp32 on DEV (template, search) pairs + network
                     latency. Also proves the build matches round 4 (calibration names, and boxes on one DEV video).
  --stage track    : DEV tracking (all 20 videos) for the given configs/draws; resumable (finished runs are skipped).
  --stage report   : combines both stages with the existing round-4 DEV runs (none, keep1) into REPORT.md.

Changes no existing code; uses hiftstress functions unchanged.
"""
import argparse
import copy
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiftstress.crops import gt_crops, gt_pairs  # noqa: E402
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.edge import _quantize_backbone_static, benchmark  # noqa: E402
from hiftstress.metrics import sequence_scores  # noqa: E402
from hiftstress.model import build_model  # noqa: E402
from hiftstress.tracker import TrackingModel, run_sequence, score_map  # noqa: E402

LAYERS = ["layer1", "layer2", "layer3", "layer4", "layer5"]
CONFIGS = {"none": ()}
CONFIGS.update({f"keep{k}": (f"layer{k}",) for k in range(1, 6)})
CONFIGS.update({f"only{k}": tuple(l for l in LAYERS if l != f"layer{k}") for k in range(1, 6)})
REUSE = {"none": "uav123-dev20_int8bb_c40x4s_cal{d}", "keep1": "uav123-dev20_int8bb-s1_c40x4s_cal{d}"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["fidelity", "track", "report"])
    p.add_argument("--snapshot", default="weights/general_model.pth")
    p.add_argument("--allow-random", action="store_true", help="smoke test only")
    p.add_argument("--dataset", default="flat")
    p.add_argument("--root", default="data/uav123_flat")
    p.add_argument("--seqs-file", default="plan/round3/dev-20.json")
    p.add_argument("--calib-exclude", default="attack-subset-40.json")
    p.add_argument("--calib-seqs", type=int, default=40)
    p.add_argument("--calib-frames", type=int, default=4)
    p.add_argument("--draws", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--fid-frames", type=int, default=10, help="spread search frames per DEV video for fidelity")
    p.add_argument("--net-iters", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0, help="smoke tests only")
    p.add_argument("--results", default="results", help="where the round-4 DEV runs and fp32 CPU DEV run live")
    p.add_argument("--fp32-ref", default="uav123-dev20_fp32_clean_nodef_t4")
    p.add_argument("--out", default="results/layer_sensitivity")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def names_from(path):
    txt = Path(path).read_text()
    return json.loads(txt)["sequences"] if str(path).endswith(".json") else txt.split()


def calibration(a, eval_names, draw):
    """Exactly scripts/run_track.py's calibration path (seed+1 pool, evaluated + excluded videos removed)."""
    pool = load_dataset(a.dataset, a.root, None, None, None, a.seed + 1)
    pool = [s for s in pool if s.name not in set(eval_names)] or pool
    ex = set(names_from(a.calib_exclude))
    pool = [s for s in pool if s.name not in ex]
    idx = np.random.default_rng(1000 + draw).choice(len(pool), min(a.calib_seqs, len(pool)), replace=False)
    cal = [pool[i] for i in sorted(set(idx))]
    _, crops = gt_crops(cal, frames_per_seq=a.calib_frames, device="cpu", spread=True)
    return [s.name for s in cal], [c.cpu() for c in crops]


def build(net_fp32, crops, keep):
    net = copy.deepcopy(net_fp32).cpu().eval()
    net, engine = _quantize_backbone_static(net, crops, keep)
    return net, engine


def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]), np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-12)


@torch.no_grad()
def fidelity(ref, net, pairs):
    same, err = [], []
    for z, x in pairs:
        ref.template(z); net.template(z)
        s0, s1 = score_map(ref, x).flatten(), score_map(net, x).flatten()
        same.append(int(s0.argmax() == s1.argmax()))
        err.append(float((s0 - s1).norm() / (s0.norm() + 1e-12)))
    return float(np.mean(same)), float(np.mean(err)), float(np.median(err))


def stage_fidelity(a, out):
    torch.set_num_threads(a.threads)
    dev_names = names_from(a.seqs_file)
    dev = load_dataset(a.dataset, a.root, None, dev_names, a.max_frames or None, a.seed)
    net_fp32, _ = build_model(None if a.allow_random else a.snapshot, torch.device("cpu"))
    net_fp32 = net_fp32.cpu().eval()
    pairs = [(z.cpu(), x.cpu()) for z, x in gt_pairs(dev, frames_per_seq=a.fid_frames, device="cpu", spread=True)]
    zb, xb = pairs[0]
    configs = a.configs or list(CONFIGS)
    rows, ident = [], {}
    fp32_lat = [benchmark(net_fp32, torch.float32, torch.device("cpu"), zb, xb, warmup=10, iters=a.net_iters)]
    for d in a.draws:
        cal_names, crops = calibration(a, dev_names, d)
        ref_sum = Path(a.results) / REUSE["keep1"].format(d=d) / "summary.json"
        if ref_sum.exists():
            saved = json.loads(ref_sum.read_text()).get("variant_info", {}).get("calibration_sequences")
            ident[f"draw{d}_calibration_matches_round4"] = saved == cal_names
            if saved != cal_names and not a.allow_random:
                sys.exit(f"STOP: draw {d} calibration differs from round 4's DEV run; not the same models.")
        for c in configs:
            t0 = time.time()
            net, engine = build(net_fp32, crops, CONFIGS[c])
            sp, em, emed = fidelity(net_fp32, net, pairs)
            lat = benchmark(net, torch.float32, torch.device("cpu"), zb, xb, warmup=10, iters=a.net_iters)
            if c in REUSE and d == a.draws[0]:      # identity: re-track the shortest DEV video, compare with round 4's boxes
                seq = min(dev, key=len)
                preds, _, _ = run_sequence(seq, TrackingModel(net, torch.device("cpu"), torch.float32), torch.device("cpu"))
                f = Path(a.results) / REUSE[c].format(d=d) / "preds" / f"{seq.name}.txt"
                if f.exists():
                    ref = np.loadtxt(f, delimiter=",", ndmin=2)[: len(preds)]
                    ident[f"{c}_draw{d}_{seq.name}_boxes_iou>0.99"] = float((iou(preds[1:], ref[1:]) > 0.99).mean())
            rows.append(dict(draw=d, config=c, fp32_layers="+".join(CONFIGS[c]) or "-", same_peak=round(sp, 4),
                             rel_err_mean=round(em, 5), rel_err_median=round(emed, 5),
                             net_ms=round(lat["latency_ms_mean"], 2), backbone_ms=round(lat["backbone_ms_mean"], 2),
                             build_s=round(time.time() - t0, 1)))
            print(f"  draw {d} {c:6s} keep fp32 {rows[-1]['fp32_layers']:34s} same-peak {sp:.3f}  rel-err {em:.4f}  net {lat['latency_ms_mean']:.1f} ms")
    fp32_lat.append(benchmark(net_fp32, torch.float32, torch.device("cpu"), zb, xb, warmup=10, iters=a.net_iters))
    with open(out / "fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "fidelity_meta.json").write_text(json.dumps(dict(
        pairs=len(pairs), dev_videos=len(dev), draws=a.draws, threads=a.threads, identity=ident,
        fp32_net_ms=[round(x["latency_ms_mean"], 2) for x in fp32_lat], engine=engine), indent=1))
    print("identity:", ident)


def stage_track(a, out):
    torch.set_num_threads(a.threads)
    dev_names = names_from(a.seqs_file)
    dev = load_dataset(a.dataset, a.root, None, dev_names, a.max_frames or None, a.seed)
    net_fp32, _ = build_model(None if a.allow_random else a.snapshot, torch.device("cpu"))
    net_fp32 = net_fp32.cpu().eval()
    configs = a.configs or ["keep2", "keep3", "keep4", "keep5"]
    for d in a.draws:
        cal_names, crops = None, None
        for c in configs:
            run = out / "track" / f"{c}_cal{d}"
            if (run / "per_seq.json").exists():
                print(f"  skip {run.name} (done)"); continue
            if crops is None:
                cal_names, crops = calibration(a, dev_names, d)
            net, engine = build(net_fp32, crops, CONFIGS[c])
            (run / "preds").mkdir(parents=True, exist_ok=True)
            per, frames, step = [], 0, 0.0
            for seq in dev:
                preds, _, times = run_sequence(seq, TrackingModel(net, torch.device("cpu"), torch.float32), torch.device("cpu"))
                np.savetxt(run / "preds" / f"{seq.name}.txt", preds, fmt="%.4f", delimiter=",")
                s = sequence_scores(preds, seq.gt); s["name"] = seq.name; per.append(s)
                frames += len(times) - 1; step += float(times[1:].sum())
            (run / "summary.json").write_text(json.dumps(dict(
                config=c, fp32_layers=list(CONFIGS[c]), draw=d, calibration_sequences=cal_names, engine=engine,
                videos=len(per), success_auc=float(np.mean([p["success_auc"] for p in per])),
                ms_per_frame=1e3 * step / max(frames, 1), threads=a.threads), indent=1))
            (run / "per_seq.json").write_text(json.dumps(per, indent=1))      # written last = run complete
            print(f"  done {run.name}: success {np.mean([p['success_auc'] for p in per]):.4f}, {1e3 * step / max(frames, 1):.1f} ms/frame")


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    b = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def stage_report(a, out):
    R = Path(a.results)
    lines = ["# Per-layer int8 sensitivity of HiFT's backbone (DEV, 20 videos)", ""]
    fid = list(csv.DictReader(open(out / "fidelity.csv"))) if (out / "fidelity.csv").exists() else []
    meta = json.loads((out / "fidelity_meta.json").read_text()) if (out / "fidelity_meta.json").exists() else {}
    if fid:
        lines += [f"Identity with round 4: {meta.get('identity')}", "",
                  "## A. Score-map fidelity vs fp32 (mean over draws)", "",
                  f"{meta.get('pairs')} DEV (template, search) pairs; same-peak = share of crops where the int8 model's "
                  "score-map peak is at the same cell as fp32; rel-err = relative L2 error of the fused score map. "
                  f"fp32 network latency {meta.get('fp32_net_ms')} ms (start, end).", "",
                  "| Config | Layers kept fp32 | Same-peak | Rel. error | Network ms |", "|---|---|---|---|---|"]
        for c in CONFIGS:
            r = [x for x in fid if x["config"] == c]
            if r:
                f = lambda k: np.mean([float(x[k]) for x in r])  # noqa: E731
                lines.append(f"| {c} | {r[0]['fp32_layers']} | {f('same_peak'):.3f} | {f('rel_err_mean'):.4f} | {f('net_ms'):.1f} |")
        only = {c: np.mean([float(x["rel_err_mean"]) for x in fid if x["config"] == c]) for c in CONFIGS if c.startswith("only")}
        if only:
            rank = sorted(only, key=lambda c: -only[c])
            lines += ["", f"**Q1. Most sensitive single layer (largest error when quantised alone):** layer{rank[0][-1]} "
                          f"(ranking: {', '.join('layer' + c[-1] for c in rank)}). "
                          f"Layer1 is {'the most' if rank[0] == 'only1' else 'NOT the most'} sensitive."]
    fp = R / a.fp32_ref / "per_seq.json"
    if fp.exists():
        base = {p["name"]: p["success_auc"] for p in json.loads(fp.read_text())}
        lines += ["", "## B. DEV tracking: change in success AUC vs fp32 CPU (per video: mean over the available calibration draws)", "",
                  "| Config | Layers kept fp32 | Draws | Change [95% CI] | Excluding person19_2 | ms/frame |", "|---|---|---|---|---|---|"]
        for c in ["none"] + [f"keep{k}" for k in range(1, 6)]:
            per_draw, ms = [], []
            for d in a.draws:
                f = (R / REUSE[c].format(d=d) / "per_seq.json") if c in REUSE else (out / "track" / f"{c}_cal{d}" / "per_seq.json")
                if f.exists():
                    per_draw.append({p["name"]: p["success_auc"] for p in json.loads(f.read_text())})
                    sj = f.parent / "summary.json"
                    if sj.exists():
                        s = json.loads(sj.read_text())
                        ms.append(s["ms_per_frame"]["total"] if isinstance(s.get("ms_per_frame"), dict) else s.get("ms_per_frame"))
            if not per_draw:
                continue
            vids = sorted(base)
            ch = [np.mean([pd[v] for pd in per_draw]) - base[v] for v in vids]
            m, lo, hi = boot(ch)
            m2, lo2, hi2 = boot([x for v, x in zip(vids, ch) if v != "person19_2"])
            lines.append(f"| {c} | {'+'.join(CONFIGS[c]) or '-'} | {len(per_draw)} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                         f"{m2:+.4f} [{lo2:+.4f}, {hi2:+.4f}] | {np.mean([x for x in ms if x]):.1f} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    {"fidelity": stage_fidelity, "track": stage_track, "report": stage_report}[a.stage](a, out)


if __name__ == "__main__":
    main()
