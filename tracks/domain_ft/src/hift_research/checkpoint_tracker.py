"""Build a HiFT tracker from the official checkpoint or a B2 domain/LaSOT checkpoint (same rules as
scripts/track_b2.py, which stays unchanged for the recorded runs): strict key loading, corrected encoder
layout unless the checkpoint records layout "original", ResNet-18 graphs via GraphTracker."""
from __future__ import annotations

from pathlib import Path
from types import MethodType

B2_SCHEMAS = ("hift-wave4-serial-pilot-boundary-v1", "hift-b2-lasot-checkpoint-v1", "hift-b2-domain-checkpoint-v1")
UPSTREAM_MANIFEST = "configs/upstream_992aa7c_files.sha256"  # git-blob sha256 of every tracked upstream file


def verify_upstream_files(root):
    """Fail-closed upstream identity for a code copy without .git (cluster bundle): every tracked file of the
    pinned upstream commit must be present with its git-blob sha256."""
    from .provenance import sha256
    root = Path(root)
    upstream = root/"third_party/HiFT"
    lines = (root/UPSTREAM_MANIFEST).read_text(encoding="utf-8").splitlines()
    bad = [name for digest, name in (line.split(" *", 1) for line in lines if line.strip())
           if not (upstream/name).is_file() or sha256(upstream/name) != digest]
    if bad or not lines:
        raise ValueError(f"upstream files differ from the pinned commit: {bad[:5]}")
    return len(lines)


TUNABLE_TRACK_KEYS = ("PENALTY_K", "WINDOW_INFLUENCE", "LR")  # stage-3 post-processing grid (REF/T identity)


def apply_track_overrides(track_cfg, overrides):
    """Set whitelisted cfg.TRACK post-processing values in place; returns the effective values of all tunable keys.
    Unknown keys, non-finite values, WINDOW_INFLUENCE outside [0, 1] and negative PENALTY_K/LR are rejected."""
    import math
    overrides = dict(overrides or {})
    unknown = set(overrides)-set(TUNABLE_TRACK_KEYS)
    if unknown:
        raise ValueError(f"not tunable: {sorted(unknown)}")
    for key, value in overrides.items():
        value = float(value)
        if not math.isfinite(value) or value < 0 or (key == "WINDOW_INFLUENCE" and value > 1):
            raise ValueError(f"invalid {key}={value}")
        track_cfg[key] = value
    return {key: float(track_cfg[key]) for key in TUNABLE_TRACK_KEYS}


def ensure_upstream_identity(root):
    """'git' when the submodule checkout exists (inference.configure_upstream verifies it unchanged); otherwise
    verify the file manifest and let configure_upstream use it instead of git (isolated adapter: inference.py
    and provenance.py stay unchanged)."""
    root = Path(root)
    if (root/"third_party/HiFT/.git").exists():
        return "git"
    from . import provenance
    count = verify_upstream_files(root)
    provenance.source_identity = lambda _root: {"upstream_verification": f"file manifest ({count} files)"}
    return f"file manifest ({count} files, {UPSTREAM_MANIFEST})"


def build_tracker(root, checkpoint=None, *, official=False, score="sigmoid", selection="nan-safe", track_overrides=None):
    """Returns (tracker, info). score: raw|sigmoid; selection: upstream|nan-safe; track_overrides: whitelisted
    cfg.TRACK values (None = upstream config). The upstream tracker reads cfg.TRACK at every frame."""
    import torch
    from .corrections.tensor_layout import encoder_forward_corrected
    from .inference import ScoreProbabilityAdapter, configure_upstream, load_original, make_tracker
    from .provenance import sha256

    root = Path(root)
    upstream_verification = ensure_upstream_identity(root)
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
    track_config = apply_track_overrides(cfg.TRACK, track_overrides)
    if selection == "nan-safe":
        from .tracking_nan_safe import make_nan_safe_tracker
        tracker = make_nan_safe_tracker(ScoreProbabilityAdapter(model) if score == "sigmoid" else model)
    elif selection == "upstream":
        tracker = make_tracker(model, "B1-score" if score == "sigmoid" else "B0")
    else:
        raise ValueError("selection must be upstream or nan-safe")
    info.update({"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
                 "score": score, "selection": selection, "upstream_verification": upstream_verification,
                 "track_config": track_config, "track_overrides": dict(track_overrides or {})})
    return tracker, info
