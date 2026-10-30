"""Model variants for the edge-cost track, and a latency / size / FLOPs benchmark.

Variants (none needs retraining):
  fp32            the reference model
  fp16            half precision (GPU / Apple MPS only)
  int8dyn         dynamic int8 quantisation of the transformer's Linear layers (CPU only)
  int8bb          static int8 quantisation of the AlexNet backbone, calibrated on real
                  search crops; the transformer head stays fp32 (CPU only)
  int8full        int8bb + static int8 of the head's convolutions (fusion convs, loc/cls
                  towers); the transformer and the template/search correlation stay fp32 (CPU only)
  prune30/50/70   global L1 unstructured pruning of conv + linear weights (any device);
                  this removes weights but dense hardware does not get faster, which is
                  itself a result worth reporting
  compile         torch.compile of the backbone and head (GPU)
"""
import copy
import io
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .model import count_params, load_cfg

cfg = load_cfg()
VARIANTS = ["fp32", "fp16", "int8dyn", "int8bb", "int8full", "prune30", "prune50", "prune70", "compile"]


def state_size_mb(module):
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def _prune(net, amount):
    import torch.nn.utils.prune as prune
    targets = [(m, "weight") for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for m, p in targets:
        prune.remove(m, p)
    return net


def _quantize_backbone_static(net, calib_crops):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    import platform
    supported = torch.backends.quantized.supported_engines
    engine = "qnnpack" if platform.machine().lower() in ("arm64", "aarch64") else "fbgemm"
    if engine not in supported:
        engine = next(e for e in ("fbgemm", "x86", "qnnpack") if e in supported)
    torch.backends.quantized.engine = engine
    bb = net.backbone.eval()
    example = (calib_crops[0],)
    prepared = prepare_fx(bb, get_default_qconfig_mapping(engine), example)
    with torch.no_grad():
        for c in calib_crops:
            prepared(c)
    net.backbone = convert_fx(prepared)
    return net, engine


def _quantize_head_static(net, calib_pairs, engine):
    """Static int8 for the head's conv layers. calib_pairs: list of (template crop, search crop)."""
    import torch.nn.functional as F
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    qm = (get_default_qconfig_mapping(engine)
          .set_module_name("transformer", None).set_module_name("row_embed", None)
          .set_module_name("col_embed", None).set_object_type(F.conv2d, None))  # keep xcorr fp32
    head = copy.deepcopy(net.grader).eval()
    with torch.no_grad():
        net.template(calib_pairs[0][0])
        example = (net.backbone(calib_pairs[0][1]), net.zf)
        prepared = prepare_fx(head, qm, example, prepare_custom_config={"non_traceable_module_name": ["transformer"]})
        for z, x in calib_pairs:
            net.template(z)
            prepared(net.backbone(x), net.zf)
    net.grader = convert_fx(prepared)
    return net


def make_variant(name, net_fp32, device, calib_crops=None, calib_pairs=None):
    """Returns (net, dtype, device, info). Never modifies net_fp32."""
    info = {"variant": name}
    if name == "fp32":
        return net_fp32, torch.float32, device, info
    if name == "fp16":
        if device.type not in ("cuda", "mps"):
            raise SystemExit("fp16 needs a GPU (cuda) or Apple MPS")
        return copy.deepcopy(net_fp32).half(), torch.float16, device, info
    if name.startswith("prune"):
        amount = int(name[len("prune"):]) / 100
        net = _prune(copy.deepcopy(net_fp32).cpu(), amount).to(device).eval()
        total, nonzero = count_params(net)
        info.update(target_sparsity=amount, nonzero_fraction=nonzero / total)
        return net, torch.float32, device, info
    if name == "int8dyn":
        net = copy.deepcopy(net_fp32).cpu().eval()
        net = torch.ao.quantization.quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)
        n_lin = sum(1 for m in net.modules() if type(m).__name__ == "Linear" and "quantized" in type(m).__module__)
        info.update(note="CPU only; MultiheadAttention projections stay fp32", quantized_linear_layers=n_lin)
        return net, torch.float32, torch.device("cpu"), info
    if name == "int8bb":
        if not calib_crops:
            raise SystemExit("int8bb needs calibration crops")
        net = copy.deepcopy(net_fp32).cpu().eval()
        net, engine = _quantize_backbone_static(net, [c.cpu() for c in calib_crops])
        info.update(note="CPU only; backbone int8, transformer head fp32", engine=engine,
                    calibration_crops=len(calib_crops))
        return net, torch.float32, torch.device("cpu"), info
    if name == "int8full":
        if not calib_pairs:
            raise SystemExit("int8full needs calibration (template, search) pairs")
        net = copy.deepcopy(net_fp32).cpu().eval()
        net, engine = _quantize_backbone_static(net, [x.cpu() for _, x in calib_pairs])
        net = _quantize_head_static(net, [(z.cpu(), x.cpu()) for z, x in calib_pairs], engine)
        info.update(note="CPU only; backbone + head convs int8; transformer and correlation fp32",
                    engine=engine, calibration_pairs=len(calib_pairs))
        return net, torch.float32, torch.device("cpu"), info
    if name == "compile":
        net = copy.deepcopy(net_fp32)
        net.backbone = torch.compile(net.backbone)
        net.grader = torch.compile(net.grader)
        return net, torch.float32, device, info
    raise SystemExit(f"unknown variant {name}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark(net, dtype, device, z, x, warmup=20, iters=200):
    """Network-only cost of one tracking step (template is computed once, as in tracking)."""
    z, x = z.to(device, dtype), x.to(device, dtype)
    net.template(z)
    for _ in range(warmup):
        net.track(x)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lat, bb_lat = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        xf = net.backbone(x)
        _sync(device)
        t1 = time.perf_counter()
        net.grader(xf, net.zf)
        _sync(device)
        t2 = time.perf_counter()
        lat.append(t2 - t0)
        bb_lat.append(t1 - t0)
    lat, bb_lat = np.array(lat) * 1e3, np.array(bb_lat) * 1e3
    out = {
        "latency_ms_mean": float(lat.mean()), "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)), "latency_ms_std": float(lat.std()),
        "backbone_ms_mean": float(bb_lat.mean()), "head_ms_mean": float((lat - bb_lat).mean()),
        "fps_network": float(1e3 / lat.mean()), "iters": iters,
    }
    if device.type == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return out


@torch.no_grad()
def count_flops(net, z, x):
    """FLOPs of one tracking step (fp32 reference; quantised variants keep the same count)."""
    net.template(z)
    with FlopCounterMode(display=False) as fc:
        xf = net.backbone(x)
    bb = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        net.grader(xf, net.zf)
    return {"gflops_backbone": bb / 1e9, "gflops_head": fc.get_total_flops() / 1e9,
            "gflops_total": (bb + fc.get_total_flops()) / 1e9}
