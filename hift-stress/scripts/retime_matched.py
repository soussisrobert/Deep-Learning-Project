"""Matched re-timing: fp32 vs the exact quality-tested int8bb-s1 model, in repeated interleaved sessions.

Why: the round-4 speed-up (1.80x network) came from bench_edge.py, which built its own int8 model with a
different calibration (20 videos, default frames) from the one whose accuracy was tested; and the fp32
tracker-step time (91.9 ms) came from a different job. This script removes both gaps:

  1. It builds int8bb-s1 with exactly the run_track.py code path and recipe of the accuracy run
     uav123-sub40_int8sel_clean_nodef_t4_cal1 (--calib-seqs 40 --calib-frames 4 --calib-spread --calib-set 1,
     calibration pool = all videos minus the 40 TEST videos, seed 0), and PROVES it is the same model:
       a. the 40 calibration video names must equal those saved in that run's summary.json;
       b. the boxes it tracks on the timing videos are compared with that run's saved boxes.
  2. fp32 and int8 run in the same process, on the same node, with the same thread count, in S sessions
     whose order alternates (fp32 first in odd sessions, int8 first in even ones).
     Each session measures, for both models:
       - network latency (backbone + head on real search crops, after warm-up), and
       - tracker step per frame on fixed TEST videos (crop + network + post-processing; image loading excluded),
     so every speed-up is a within-session ratio.

  python scripts/retime_matched.py --snapshot weights/general_model.pth --root data/uav123_flat \
      --ref-int8 results/uav123-sub40_int8sel_clean_nodef_t4_cal1 --ref-fp32 results/uav123_fp32_clean_nodef_t4

Writes results/retime_matched/{summary.json,sessions.csv,REPORT.md}. Changes nothing else.
"""
import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hiftstress.crops import gt_crops, gt_pairs  # noqa: E402
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.edge import benchmark, make_variant, state_size_mb  # noqa: E402
from hiftstress.metrics import sequence_scores  # noqa: E402
from hiftstress.model import build_model  # noqa: E402
from hiftstress.tracker import TrackingModel, run_sequence  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", default="flat")
    p.add_argument("--root", default="data/uav123_flat")
    p.add_argument("--seqs-file", default="attack-subset-40.json", help="evaluation set of the accuracy run (TEST)")
    p.add_argument("--variant", default="int8bb-s1")
    p.add_argument("--calib-seqs", type=int, default=40)
    p.add_argument("--calib-frames", type=int, default=4)
    p.add_argument("--no-calib-spread", action="store_true")
    p.add_argument("--calib-set", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, nargs="+", default=[4], help="thread counts; each gets its own sessions")
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--time-seqs", nargs="+", default=["bike1", "car1_2", "person11", "person7_2"],
                   help="TEST videos used for the tracker-step timing (also used for the identity check)")
    p.add_argument("--max-frames", type=int, default=0, help="limit frames per timing video (0 = all; for smoke tests)")
    p.add_argument("--net-iters", type=int, default=300)
    p.add_argument("--net-crops", type=int, default=8, help="real search crops cycled in the network timing")
    p.add_argument("--ref-int8", help="results folder of the accuracy-tested int8 run (identity check)")
    p.add_argument("--ref-fp32", help="results folder of the fp32 CPU accuracy run (consistency check)")
    p.add_argument("--allow-random", action="store_true", help="smoke test only: random weights, no identity check")
    p.add_argument("--out", default="results/retime_matched")
    return p.parse_args()


def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]), np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-12)


def compare(preds, ref_dir, name):
    f = Path(ref_dir) / "preds" / f"{name}.txt"
    if not f.exists():
        return None
    ref = np.loadtxt(f, delimiter=",", ndmin=2)[: len(preds)]
    o = iou(preds[1:], ref[1:])
    return dict(frames=int(len(o)), same_box_iou_0_99=float((o > 0.99).mean()), median_iou=float(np.median(o)),
                max_abs_px=float(np.abs(preds[1:] - ref[1:]).max()))


def cpu_name():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        return next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), platform.processor())
    except Exception:  # noqa: BLE001
        return platform.processor()


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    snap = None if a.allow_random else a.snapshot

    # ---- evaluation set and calibration exactly as scripts/run_track.py does it
    txt = Path(a.seqs_file).read_text()
    test_names = json.loads(txt)["sequences"] if a.seqs_file.endswith(".json") else txt.split()
    seqs = load_dataset(a.dataset, a.root, None, test_names, None, a.seed)
    net_fp32, _ = build_model(snap, device)
    pool = load_dataset(a.dataset, a.root, None, None, None, a.seed + 1)
    evald = {s.name for s in seqs}
    pool = [s for s in pool if s.name not in evald] or pool
    if a.calib_set:
        idx = np.random.default_rng(1000 + a.calib_set).choice(len(pool), min(a.calib_seqs, len(pool)), replace=False)
    else:
        idx = np.linspace(0, len(pool) - 1, num=min(a.calib_seqs, len(pool))).round().astype(int)
    cal_seqs = [pool[i] for i in sorted(set(idx))]
    cal_names = [s.name for s in cal_seqs]
    spread = not a.no_calib_spread
    _, calib = gt_crops(cal_seqs, frames_per_seq=a.calib_frames, device="cpu", spread=spread)
    pairs = gt_pairs(cal_seqs, frames_per_seq=a.calib_frames, device="cpu", spread=spread)
    net_q, dtype_q, dev_q, info_q = make_variant(a.variant, net_fp32, device, calib, pairs)
    net_fp32 = net_fp32.to(device).eval()

    identity = {"calibration_sequences_match": None}
    if a.ref_int8:
        ref_sum = json.loads((Path(a.ref_int8) / "summary.json").read_text())
        ref_names = ref_sum.get("variant_info", {}).get("calibration_sequences")
        identity["calibration_sequences_match"] = (ref_names == cal_names)
        identity["ref_run"] = ref_sum.get("run")
        identity["ref_calibration_crops"] = ref_sum.get("variant_info", {}).get("calibration_crops")
        identity["this_calibration_crops"] = info_q.get("calibration_crops")
        if not identity["calibration_sequences_match"] and not a.allow_random:
            sys.exit("STOP: calibration videos differ from the accuracy-tested run; this is not the same model.")
    print(f"[retime] int8 model: {a.variant}, calibration {len(cal_names)} videos x {a.calib_frames} "
          f"({'spread' if spread else 'first frames'}), set {a.calib_set}; identity: {identity}")

    tseqs = [s for s in seqs if s.name in set(a.time_seqs)]
    if a.max_frames:
        for s in tseqs:
            s.frames, s.gt = s.frames[: a.max_frames], s.gt[: a.max_frames]
    _, xs = gt_crops(tseqs, frames_per_seq=max(1, a.net_crops // max(1, len(tseqs))), device="cpu", spread=True)
    zs, _ = gt_crops(tseqs[:1], frames_per_seq=1, device="cpu")
    models = {"fp32": (net_fp32, torch.float32), a.variant: (net_q, dtype_q)}
    sizes = {"fp32": state_size_mb(net_fp32), a.variant: state_size_mb(net_q)}

    rows, checks = [], {}
    for th in a.threads:
        torch.set_num_threads(th)
        for s in range(1, a.sessions + 1):
            order = ["fp32", a.variant] if s % 2 else [a.variant, "fp32"]
            for m in order:
                net, dt = models[m]
                # network latency, cycling real crops so one easy crop cannot bias it
                nb = [benchmark(net, dt, device, zs[0], x, warmup=10, iters=max(1, a.net_iters // len(xs))) for x in xs]
                n_iter = sum(r["iters"] for r in nb)
                net_ms = sum(r["latency_ms_mean"] * r["iters"] for r in nb) / n_iter
                bb_ms = sum(r["backbone_ms_mean"] * r["iters"] for r in nb) / n_iter
                # tracker step on the fixed timing videos (image loading excluded by run_sequence)
                frames, step_s, model_s = 0, 0.0, 0.0
                for seq in tseqs:
                    tm = TrackingModel(net, device, dt)
                    preds, _, times = run_sequence(seq, tm, device)
                    frames += len(times) - 1
                    step_s += float(times[1:].sum())
                    model_s += tm.stats["model_s"]
                    if th == a.threads[0] and s <= 2:       # identity / consistency check on the first sessions
                        ref = a.ref_int8 if m == a.variant else a.ref_fp32
                        if ref:
                            checks.setdefault(f"{m}_session{s}", {})[seq.name] = compare(preds, ref, seq.name)
                        checks.setdefault(f"{m}_auc_session{s}", {})[seq.name] = sequence_scores(preds, seq.gt)["success_auc"]
                rows.append(dict(threads=th, session=s, position=order.index(m) + 1, model=m,
                                 net_ms=round(net_ms, 3), backbone_ms=round(bb_ms, 3), head_ms=round(net_ms - bb_ms, 3),
                                 step_ms=round(1e3 * step_s / frames, 3), model_ms=round(1e3 * model_s / frames, 3),
                                 frames=frames))
                print(f"  threads {th} session {s} {m:10s} net {net_ms:7.2f} ms  step {1e3 * step_s / frames:7.2f} ms/frame")

    with open(out / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def ratios(th, key):
        r = []
        for s in range(1, a.sessions + 1):
            f = next(x for x in rows if x["threads"] == th and x["session"] == s and x["model"] == "fp32")[key]
            q = next(x for x in rows if x["threads"] == th and x["session"] == s and x["model"] == a.variant)[key]
            r.append(f / q)
        return r

    summ = dict(cpu=cpu_name(), host=socket.gethostname(), torch=torch.__version__, sessions=a.sessions,
                time_seqs=[s.name for s in tseqs], frames_per_session=rows[0]["frames"], variant=a.variant,
                recipe=dict(calib_seqs=a.calib_seqs, calib_frames=a.calib_frames, spread=spread, calib_set=a.calib_set,
                            calibration_sequences=cal_names),
                identity=identity, checks=checks, size_mb=sizes,
                env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_JOB_ID")}, by_threads={})
    for th in a.threads:
        d = {}
        for key in ("net_ms", "backbone_ms", "head_ms", "step_ms"):
            for m in ("fp32", a.variant):
                v = [x[key] for x in rows if x["threads"] == th and x["model"] == m]
                d[f"{m}_{key}"] = dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)))
            if key in ("net_ms", "step_ms", "backbone_ms"):
                r = ratios(th, key)
                d[f"speedup_{key}"] = dict(median=float(np.median(r)), min=float(min(r)), max=float(max(r)),
                                           per_session=[round(x, 3) for x in r])
        summ["by_threads"][str(th)] = d
    (out / "summary.json").write_text(json.dumps(summ, indent=1))

    lines = ["# Matched re-timing: fp32 vs the accuracy-tested int8 model", "",
             f"- CPU: {summ['cpu']} (host {summ['host']}), torch {summ['torch']}",
             f"- int8 model: {a.variant}, calibration {a.calib_seqs} videos x {a.calib_frames} "
             f"({'spread' if spread else 'first'}), set {a.calib_set}",
             f"- Same model as the accuracy run? calibration videos identical: **{identity['calibration_sequences_match']}**",
             f"- {a.sessions} sessions per thread count, order alternating; tracker step on {', '.join(summ['time_seqs'])} "
             f"({summ['frames_per_session']} frames per model per session; image loading excluded)", "",
             "| Threads | fp32 network ms | int8 network ms | Network speed-up | fp32 step ms | int8 step ms | Tracker-step speed-up |",
             "|---|---|---|---|---|---|---|"]
    for th, d in summ["by_threads"].items():
        g = lambda k: d[k]["median"]  # noqa: E731
        sn, ss = d["speedup_net_ms"], d["speedup_step_ms"]
        lines.append(f"| {th} | {g('fp32_net_ms'):.1f} | {g(a.variant + '_net_ms'):.1f} | **{sn['median']:.2f}×** "
                     f"({sn['min']:.2f}–{sn['max']:.2f}) | {g('fp32_step_ms'):.1f} | {g(a.variant + '_step_ms'):.1f} | "
                     f"**{ss['median']:.2f}×** ({ss['min']:.2f}–{ss['max']:.2f}) |")
    lines += ["", "Values are medians over sessions; speed-ups are within-session ratios (median, range).",
              f"Model size: fp32 {sizes['fp32']:.1f} MB, int8 {sizes[a.variant]:.1f} MB.", "",
              "Identity/consistency check (share of frames whose box matches the saved accuracy run at IoU > 0.99):"]
    for k, v in checks.items():
        if "_auc_" not in k:
            lines.append(f"- {k}: " + ", ".join(f"{n} {c['same_box_iou_0_99']:.3f}" for n, c in v.items() if c))
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
