"""Load the (patched) official HiFT model on any device."""
import os
import sys
from pathlib import Path

import torch

PKG_ROOT = Path(__file__).resolve().parents[1]
HIFT_ROOT = Path(os.environ.get("HIFT_ROOT", PKG_ROOT / "third_party" / "HiFT")).resolve()
if not (HIFT_ROOT / "pysot").is_dir():
    raise SystemExit(f"HiFT code not found at {HIFT_ROOT}. Run setup_local.sh (or set HIFT_ROOT).")
if str(HIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFT_ROOT))

from pysot.core.config import cfg  # noqa: E402

CONFIG = HIFT_ROOT / "experiments" / "config.yaml"
_cfg_loaded = False


def load_cfg():
    global _cfg_loaded
    if not _cfg_loaded:
        cfg.merge_from_file(str(CONFIG))
        _cfg_loaded = True
    return cfg


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_load(path):
    """torch.load without running pickled code. The official HiFT checkpoint stores the
    epoch as a numpy scalar, so numpy's scalar/dtype reconstructors are allow-listed."""
    import numpy as np
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"), np.dtype]
    allowed += [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_model(snapshot=None, device="cpu"):
    """Build HiFT. snapshot=None gives random weights (pipeline tests only)."""
    load_cfg()
    from pysot.models.model_builder import ModelBuilder
    torch.manual_seed(0)
    model = ModelBuilder()
    info = {"snapshot": str(snapshot) if snapshot else None}
    if snapshot:
        state = safe_load(snapshot)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        info.update(missing=missing, unexpected=list(unexpected))
        if missing:
            raise SystemExit(f"Snapshot is missing {len(missing)} weights, e.g. {missing[:5]}")
    else:
        info["warning"] = "RANDOM WEIGHTS - pipeline test only, accuracy is meaningless"
    model.eval().to(device)
    return model, info


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero
