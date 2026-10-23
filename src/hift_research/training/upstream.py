"""Load pinned class bodies in isolation; never import or modify upstream config/CUDA code."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from hift_research import UPSTREAM_COMMIT

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party/HiFT"


def source_class(relative, name, namespace, *, device_indices=False):
    path = UPSTREAM / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if device_indices:
        # Exactly the two unconditional positional-index .cuda() calls are changed.
        changes = 0
        class DeviceIndices(ast.NodeTransformer):
            def visit_Call(self, call):
                nonlocal changes
                if (isinstance(call.func, ast.Attribute) and call.func.attr == "cuda"
                        and isinstance(call.func.value, ast.Call)
                        and ast.unparse(call.func.value.func) == "t.arange"):
                    changes += 1
                    inner = call.func.value
                    inner.keywords.append(ast.keyword(arg="device", value=ast.parse(
                        "self.col_embed.weight.device", mode="eval").body))
                    return ast.copy_location(inner, call)
                return self.generic_visit(call)
        node = DeviceIndices().visit(node)
        if changes != 2:
            raise RuntimeError("pinned positional-index shim no longer matches exactly two calls")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def classes():
    path = UPSTREAM / "pysot/models/utile/tran.py"
    spec = importlib.util.spec_from_file_location("_hift_training_pinned_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    alex = source_class("pysot/models/backbone/newalexnet.py", "AlexNet", {"nn": nn})
    hift = source_class("pysot/models/utile/utile.py", "HiFT", {
        "nn": nn, "t": torch, "F": torch.nn.functional, "Transformer": module.Transformer,
    }, device_indices=True)
    iou = source_class("pysot/models/loss.py", "IOULoss", {"nn": nn, "torch": torch})
    return alex, hift, iou


def anchor_target():
    cfg = SimpleNamespace(TRAIN=SimpleNamespace(
        SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4,
    ))
    return source_class("pysot/datasets/anchortarget.py", "AnchorTarget", {
        "np": np, "cfg": cfg,
    })()
