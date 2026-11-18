"""Track a B2/B2-L1 training checkpoint on GT-free sequence inputs with the upstream HiFTTracker.

Weights: strict load of the boundary file's B2TrainingGraph state into upstream ModelBuilder
(same backbone./grader. keys), with the B2 corrected encoder layout bound to every encoder layer.
--score raw keeps upstream score mixing; --score sigmoid applies the B1-score cls2 sigmoid.
Ground truth never enters this script; evaluate with scripts/evaluate.py afterwards.

    python scripts/track_b2.py --checkpoint artifacts/b2-direct/B2-L1-001/update-0100.pt \
        --inputs artifacts/wave3-evaluation/INPUTS-001 --output artifacts/b2-eval/<id> --score sigmoid
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="B2 checkpoint (omit with --official)")
    parser.add_argument("--official", action="store_true", help="B0: hash-pinned released checkpoint, original layout")
    parser.add_argument("--inputs", type=Path, required=True, help="Directory of GT-free <sequence>.json inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", choices=["raw", "sigmoid"], required=True)
    parser.add_argument("--selection", choices=["upstream", "nan-safe"], default="upstream",
                        help="nan-safe: never select NaN/inf penalized cells (src/hift_research/tracking_nan_safe.py)")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from hift_research.corrections.tensor_layout import encoder_forward_corrected
    from hift_research.inference import load_original, make_tracker
    from hift_research.provenance import sha256
    from run_sequence import read_sequence

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    if args.official:
        from hift_research import OFFICIAL_CHECKPOINT_SHA256
        args.checkpoint = ROOT/"artifacts/checkpoints/general_model.pth"
        if sha256(args.checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("official checkpoint hash mismatch")
        state = {"schema": "official", "update": None}
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.official:
        model = load_original(ROOT, args.checkpoint, ROOT/"third_party/HiFT/experiments/config.yaml")
        from pysot.core.config import cfg
    elif state.get("schema") not in ("hift-wave4-serial-pilot-boundary-v1", "hift-b2-lasot-checkpoint-v1", "hift-b2-domain-checkpoint-v1"):
        raise ValueError("expected a B2 serial-pilot boundary, B2 LaSOT or B2 domain checkpoint")
    elif state.get("backbone", "alexnet") == "resnet18":
        from hift_research.backbones.resnet import B2ResNetGraph, GraphTracker
        from hift_research.inference import configure_upstream
        cfg = configure_upstream(ROOT, ROOT/"third_party/HiFT/experiments/config.yaml")
        graph = B2ResNetGraph()
        graph.load_state_dict(state["model"], strict=True)
        model = GraphTracker(graph.cuda()).eval()
    else:
        model = load_original(ROOT, None, ROOT/"third_party/HiFT/experiments/config.yaml")
        from pysot.core.config import cfg
        model.load_state_dict(state["model"], strict=True)
        if state.get("layout", "corrected") == "corrected":  # B0-FT original-layout checkpoints keep upstream computation
            for layer in model.grader.transformer.encoder.layers:
                layer.forward = MethodType(encoder_forward_corrected, layer)
        model.eval()
    if cfg.TRAIN.groupchannel != 32:
        raise ValueError("GroupNorm groups differ from the B2 training graph")
    if args.selection == "nan-safe":
        from hift_research.inference import ScoreProbabilityAdapter
        from hift_research.tracking_nan_safe import make_nan_safe_tracker
        tracker = make_nan_safe_tracker(ScoreProbabilityAdapter(model) if args.score == "sigmoid" else model)
    else:
        tracker = make_tracker(model, "B1-score" if args.score == "sigmoid" else "B0")

    manifest = {"schema": "hift-b2-tracking-run-v1", "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
                "checkpoint_update": state["update"], "score": args.score, "selection": args.selection,
                "layout": "original upstream layout" if args.official or state.get("layout") == "original" else "B2 corrected encoder layout",
                "inputs": str(args.inputs.resolve()), "seed": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
                "software": {"torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()},
                "gpu": torch.cuda.get_device_name(0), "sequences": {}}
    for path in sorted(p for p in args.inputs.glob("*.json") if p.name != "input_receipt.json"):
        sequence, frames = read_sequence(path.resolve())
        predictions, start = [], time.perf_counter()
        for index, frame in enumerate(frames):
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode {frame}")
            if index == 0:
                bbox = sequence["initial_bbox"]
                tracker.init(image, bbox)
            else:
                bbox = tracker.track(image)["bbox"]
            if not np.isfinite(bbox).all() or np.any(np.asarray(bbox)[2:] <= 0):
                raise ValueError(f"invalid prediction {sequence['name']} frame {index}")
            predictions.append(list(map(float, bbox)))
        csv = output/f"{sequence['name']}.csv"
        np.savetxt(csv, np.asarray(predictions), delimiter=",", fmt="%.10g")
        manifest["sequences"][sequence["name"]] = {"input_sha256": sha256(path), "frames": len(frames),
                                                  "prediction_sha256": sha256(csv), "seconds": time.perf_counter()-start}
        print(sequence["name"], len(frames), f"{time.perf_counter()-start:.1f}s", flush=True)
    if args.selection == "nan-safe":
        kind = type(tracker)
        manifest["nan_safe_counts"] = {"tracked_frames": kind.frames, "nonfinite_cells": kind.nonfinite_cells,
                                       "frames_where_upstream_would_select_nonfinite": kind.first_nan_would_have_won}
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
