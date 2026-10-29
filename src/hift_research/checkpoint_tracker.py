"""Build a HiFT tracker from the official checkpoint or a B2 domain/LaSOT checkpoint (same rules as
scripts/track_b2.py, which stays unchanged for the recorded runs): strict key loading, corrected encoder
layout unless the checkpoint records layout "original", ResNet-18 graphs via GraphTracker."""
from __future__ import annotations

from pathlib import Path
from types import MethodType

B2_SCHEMAS = ("hift-wave4-serial-pilot-boundary-v1", "hift-b2-lasot-checkpoint-v1", "hift-b2-domain-checkpoint-v1")


def build_tracker(root, checkpoint=None, *, official=False, score="sigmoid", selection="nan-safe"):
    """Returns (tracker, info). score: raw|sigmoid; selection: upstream|nan-safe."""
    import torch
    from .corrections.tensor_layout import encoder_forward_corrected
    from .inference import ScoreProbabilityAdapter, configure_upstream, load_original, make_tracker
    from .provenance import sha256

    root = Path(root)
    config = root/"third_party/HiFT/experiments/config.yaml"
    if official == (checkpoint is not None):
        raise ValueError("give exactly one of checkpoint / official")
    if official:
        from . import OFFICIAL_CHECKPOINT_SHA256
        checkpoint = root/"artifacts/checkpoints/general_model.pth"
        if sha256(checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("official checkpoint hash mismatch")
        model = load_original(root, checkpoint, config)
        info = {"layout": "original upstream layout", "update": None, "backbone": "alexnet"}
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("schema") not in B2_SCHEMAS:
            raise ValueError("expected a B2 checkpoint schema")
        if state.get("backbone", "alexnet") == "resnet18":
            from .backbones.resnet import B2ResNetGraph, GraphTracker
            configure_upstream(root, config)
            graph = B2ResNetGraph()
            graph.load_state_dict(state["model"], strict=True)
            model = GraphTracker(graph.cuda()).eval()
            layout = "B2 corrected encoder layout"
        else:
            model = load_original(root, None, config)
            model.load_state_dict(state["model"], strict=True)
            layout = state.get("layout", "corrected")
            if layout == "corrected":
                for layer in model.grader.transformer.encoder.layers:
                    layer.forward = MethodType(encoder_forward_corrected, layer)
            layout = "B2 corrected encoder layout" if layout == "corrected" else "original upstream layout"
            model.eval()
        info = {"layout": layout, "update": state.get("update"), "backbone": state.get("backbone", "alexnet")}
    from pysot.core.config import cfg
    if cfg.TRAIN.groupchannel != 32:
        raise ValueError("GroupNorm groups differ from the B2 training graph")
    if selection == "nan-safe":
        from .tracking_nan_safe import make_nan_safe_tracker
        tracker = make_nan_safe_tracker(ScoreProbabilityAdapter(model) if score == "sigmoid" else model)
    elif selection == "upstream":
        tracker = make_tracker(model, "B1-score" if score == "sigmoid" else "B0")
    else:
        raise ValueError("selection must be upstream or nan-safe")
    info.update({"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
                 "score": score, "selection": selection})
    return tracker, info
