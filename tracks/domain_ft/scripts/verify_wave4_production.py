"""Exactly FOUR synthetic SGD updates on the SAME serial future-pilot controller.

Full2 versus part1 + midwindow fault(no SGD) + restored1. No real input or
pretrained tensor is optimized. Requires exclusive lead GPU token for CUDA.
"""
import argparse
import copy
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hift_research.provenance import sha256, source_identity
from hift_research.training.graph import B2TrainingGraph
from hift_research.training.recipe import ROOT
from hift_research.training.wave4_lifecycle import cpu_tree, device_rng, seed_device
from hift_research.training.wave4_loader import SyntheticCandidateDataset, dataset_identity
from hift_research.training.wave4_pilot_engine import SerialPilotController
from hift_research.training.wave4_recipes import validate_frozen_pilot


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a == b


def run(output, *, device="cuda", gpu_token=None, config_path=ROOT/"configs/B2-Pilot-W4.json", channels_last_arrays=False):
    device = torch.device(device)
    if device.type == "cuda" and not gpu_token:
        raise PermissionError("exclusive lead GPU token required before ANY CUDA")
    config = json.loads(Path(config_path).read_text())
    validate_frozen_pilot(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    identity = {"config_sha256": sha256(config_path), "driver_sha256": sha256(__file__), "proof": "4updates shared serial controller"}
    dataset = SyntheticCandidateDataset(length=32, channels_last_arrays=channels_last_arrays)
    def controller():
        return SerialPilotController(B2TrainingGraph(), dataset, config_identity=identity,
                  data_identity={"kind": "synthetic", **dataset_identity(dataset)}, device=device, gpu_token=gpu_token,
                  synthetic=True, physical=4, attempts_per_window=16, max_updates=2, max_attempts=32)
    def collect_step(value):
        result = value.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["measurement"]["post_return_memory"] = {
                "allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": torch.cuda.max_memory_reserved(device)}
        return result
    seed_device(20261023, device, gpu_token=gpu_token)
    full = controller()
    source = source_identity(ROOT)
    critical = dict(full.code_identity)
    for relative in ("scripts/verify_wave4_production.py", "scripts/run_b2_pilot.py", "configs/B2-Pilot-W4.json", "configs/B2-Reference-W4.json"):
        critical[relative] = sha256(ROOT/relative)
    for relative, expected in critical.items():
        target = output/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT/relative).read_bytes())
        if sha256(target) != expected:
            raise ValueError(f"executing source changed during snapshot:{relative}")
    manifest = {"schema": "hift-wave4-production-controller-synthetic-verification-v1",
                "started_utc": datetime.now(timezone.utc).isoformat(), "source": source, "critical_source_sha256": critical,
                "config_identity": identity, "seed": 20261023, "device": str(device), "gpu_token": gpu_token,
                "authorization": "EXACT4 synthetic effective SGD updates ONLY; all random weights; no real images/pretrained optimization",
                "planned_effective_updates": 4, "physical": 4, "attempts_per_effective_window": 16,
                "real_sgd": False, "epoch": 0, "source_grader_lr": 0.0025,
                "synthetic_input_layout": "HWC-contiguous NumPy transposed CHW; collated channels-last-strided NCHW" if channels_last_arrays else "contiguous synthetic CHW",
                "layout_bridge": "shared CPU canonicalize_model_inputs before H2D, logical pixels unchanged",
                "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform()}}
    write_json(output/"manifest.start.json", manifest)
    write_json(output/"pilot-config.snapshot.json", config)
    collect_step(full); collect_step(full)
    expected = {"model": cpu_tree(full.model.state_dict()), "optimizer": cpu_tree(full.optimizer.state_dict()),
                "scheduler": cpu_tree(full.scheduler.state_dict()), "trace": copy.deepcopy(full.semantic_trace),
                "gradients": cpu_tree({n:p.grad for n,p in full.model.named_parameters()}), "rng": device_rng(device),
                "cursor": full.cursor, "update": full.update, "window": full.window}
    full_measurements = copy.deepcopy(full.measurements)
    del full
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_device(20261023, device, gpu_token=gpu_token)
    part = controller()
    collect_step(part)
    first_measurement = copy.deepcopy(part.measurements[0])
    part.sync()
    checkpoint_started = time.perf_counter()
    checkpoint = part.save_boundary(output/"update1-boundary.pt")
    part.sync()
    checkpoint_seconds = time.perf_counter()-checkpoint_started
    params_before_fault = cpu_tree(dict(part.model.named_parameters()))
    crash_seen = taint_step_rejected = taint_save_rejected = False
    try:
        part.step(crash_after_micro=2)
    except RuntimeError as exc:
        crash_seen = "injected midwindow" in str(exc)
    no_sgd_on_crash = equal_tree(params_before_fault, dict(part.model.named_parameters())) and part.update == 1 and part.cursor == 16
    try:
        part.step()
    except RuntimeError as exc:
        taint_step_rejected = "tainted" in str(exc)
    try:
        part.save_boundary(output/"FORBIDDEN-tainted.pt")
    except RuntimeError as exc:
        taint_save_rejected = "tainted" in str(exc)
    del part
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restored = controller()
    restored.sync()
    restore_started = time.perf_counter()
    restored.restore_boundary(checkpoint["path"], expected_sha256=checkpoint["sha256"])
    restored.sync()
    restore_seconds = time.perf_counter()-restore_started
    collect_step(restored)
    comparisons = {
        "parameters_bn_buffers_bitexact": equal_tree(expected["model"], restored.model.state_dict()),
        "momentum_optimizer_bitexact": equal_tree(expected["optimizer"], restored.optimizer.state_dict()),
        "scheduler_epoch_lr_bitexact": equal_tree(expected["scheduler"], restored.scheduler.state_dict()),
        "clipped_gradients_bitexact": equal_tree(expected["gradients"], {n:p.grad for n,p in restored.model.named_parameters()}),
        "python_numpy_torch_cpu_cuda_rng_bitexact": equal_tree(expected["rng"], device_rng(device)),
        "semantic_attempt_loss_trace_bitexact": equal_tree(expected["trace"], restored.semantic_trace),
        "committed_cursor_update_window_bitexact": (expected["cursor"],expected["update"],expected["window"]) == (restored.cursor,restored.update,restored.window),
        "midwindow_crash_injected": crash_seen, "midwindow_no_parameter_or_sgd_commit": no_sgd_on_crash,
        "tainted_step_rejected": taint_step_rejected, "tainted_checkpoint_rejected": taint_save_rejected,
        "no_tainted_checkpoint_published": not (output/"FORBIDDEN-tainted.pt").exists(),
        "all_four_expected_effective_updates": restored.update == 2 and expected["update"] == 2,
    }
    evidence = {"comparisons": comparisons, "checkpoint": checkpoint, "checkpoint_serialization_seconds": checkpoint_seconds,
                "checkpoint_restore_seconds": restore_seconds, "uninterrupted_measurements": full_measurements,
                "interrupted_first_update_measurement": first_measurement, "restored_second_update_measurement": restored.measurements[0],
                "semantic_trace": restored.semantic_trace, "device_identity": restored.device_identity,
                "bn_modes": {n:m.training for n,m in restored.model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)},
                "limitations": "exact serial boundary replay, not async-prefetch/multiworker resume; synthetic inputs, no real SGD"}
    write_json(output/"verification.json", evidence)
    if not all(comparisons.values()):
        raise AssertionError(f"shared production controller drift:{comparisons}")
    manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter()-started,
                     "result": "PASS same-production-controller synthetic CUDA boundary replay", "actual_effective_updates": 4,
                     "comparisons": comparisons, "checkpoint": checkpoint,
                     "artifacts_sha256": {str(p.relative_to(output)):sha256(p) for p in output.rglob("*") if p.is_file()}})
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"output":str(output),"updates":4,"result":manifest["result"],"wall_seconds":manifest["wall_seconds"],
                      "checkpoint_save_s":checkpoint_seconds,"checkpoint_restore_s":restore_seconds,"comparisons":comparisons},indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-token")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/B2-Pilot-W4.json")
    parser.add_argument("--channels-last-arrays", action="store_true", help="regress authentic HWC->CHW NumPy/collate layout without real images")
    args = parser.parse_args()
    run(args.output, device=args.device, gpu_token=args.gpu_token, config_path=args.config, channels_last_arrays=args.channels_last_arrays)
