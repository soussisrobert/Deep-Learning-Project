"""Bounded synthetic SGD timing and immutable evidence, excluding any data pipeline."""
import gc
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hift_research.provenance import git_output, sha256, source_identity
from .graph import B2TrainingGraph, gradient_audit, optimizer_for, synthetic_batch
from .upstream import ROOT


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def memory(device):
    if device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "note": "CUDA allocator metrics not applicable on CPU; RSS not measured"}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def summarize(values):
    return {"median_ms": float(np.median(values))*1000,
            "mean_ms": float(np.mean(values))*1000,
            "min_ms": min(values)*1000, "max_ms": max(values)*1000}


def profile_batch(batch_size, device, *, seed, warmup, repeats):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = B2TrainingGraph().to(device).train()
    optimizer = optimizer_for(model)
    # Explicit positive + negative backward probes precede timed optimization.
    probes = {}
    for negative in (False, True):
        data = synthetic_batch(batch_size, device, negative=[negative]*batch_size)
        optimizer.zero_grad(set_to_none=True)
        losses = model(*data)
        if not all(bool(torch.isfinite(loss)) for loss in losses.values()):
            raise RuntimeError("nonfinite probe loss")
        losses["total_loss"].backward()
        audit = gradient_audit(model)
        if audit["unexpected_missing"] or audit["nonfinite"]:
            raise RuntimeError(f"invalid probe gradients: {audit}")
        if not negative and audit["unexpected_zero"]:
            raise RuntimeError(f"positive probe has zero active parameter gradients: {audit}")
        if negative and float(losses["loc_loss"].detach()) != 0:
            raise RuntimeError("known-negative localization loss must be zero")
        probes["negative" if negative else "positive"] = {
            "losses": {key: float(loss.detach()) for key, loss in losses.items()}, "gradients": audit}
    del data, losses
    optimizer.zero_grad(set_to_none=True)
    # B1 alternates positive/negative resident minibatches; B2/4 contain a 50/50 mix.
    batches = [synthetic_batch(batch_size, device)]
    if batch_size == 1:
        batches.append(synthetic_batch(1, device, negative=[True]))
    samples, records = [], []
    for iteration in range(warmup + repeats):
        if iteration == warmup:
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        timings = {}
        def stage(name, operation):
            synchronize(device)
            start = time.perf_counter()
            result = operation()
            synchronize(device)
            timings[name] = time.perf_counter()-start
            return result
        template, search, targets = batches[iteration % len(batches)]
        synchronize(device)
        start = time.perf_counter()
        stage("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
        z = stage("template_backbone", lambda: model.backbone(template))
        x = stage("search_backbone", lambda: model.backbone(search))
        heads = stage("correlation_transformer_heads", lambda: model.grader(x, z))
        losses = stage("decode_losses", lambda: model.losses(heads, targets))
        stage("backward", lambda: losses["total_loss"].backward())
        norm = stage("gradient_clip", lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        stage("sgd_step", optimizer.step)
        timings["total"] = time.perf_counter()-start
        if not all(bool(torch.isfinite(v)) for v in losses.values()) or not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite timed loss/gradient norm")
        record = {"iteration": iteration, "losses": {key: float(value.detach()) for key, value in losses.items()},
                  "preclip_gradient_norm": float(norm.detach()), "seconds": timings,
                  "phase": "warmup" if iteration < warmup else "timed",
                  "negative_flags": ([bool(iteration % 2)] if batch_size == 1
                                     else [bool(i % 2) for i in range(batch_size)])}
        records.append(record)
        if iteration >= warmup:
            samples.append(timings)
        # Release references before next forward; no accidental retained previous graph.
        del z, x, heads, losses
    audit = gradient_audit(model)
    if audit["unexpected_missing"] or audit["nonfinite"]:
        raise RuntimeError(f"invalid final gradients: {audit}")
    stages = {key: summarize([sample[key] for sample in samples]) for key in samples[0]}
    total_seconds = sum(sample["total"] for sample in samples)
    return {"status": "measured", "batch_size": batch_size, "probes": probes,
            "iterations": records, "stage_latency": stages, "final_gradients": audit,
            "synthetic_pairs_per_second": batch_size*len(samples)/total_seconds,
            "memory": memory(device), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "limitations": ["resident synthetic crops/targets; no I/O, augmentation, loading, H2D, validation or checkpoint saves",
                            "synchronized per-stage wall time includes launch/synchronization overhead",
                            "negative-only iterations deliberately give zero localization-head gradients",
                            "few random-initialized SGD steps are feasibility, not trained accuracy or convergence"]}


def run_profile(output, *, device="cpu", batches=(1, 2, 4), seed=20261023,
                warmup=2, repeats=5, threads=1, gpu_token=None):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU/CUDA supported")
    if device.type == "cuda" and not gpu_token:
        raise ValueError("CUDA requires explicit lead-granted --gpu-token")
    if warmup < 1 or repeats < 5 or threads < 1:
        raise ValueError("require warmup>=1 repeats>=5 threads>=1")
    if not batches or any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in batches):
        raise ValueError("batches must be positive integers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    identity = source_identity(ROOT)
    # Freeze exact content, including uncommitted adapters. Other workers may edit
    # unrelated files concurrently; source hashes and copied contents must agree.
    sources = output / "sources"
    sources.mkdir()
    for relative, expected in identity["source_files_sha256"].items():
        target = sources / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if sha256(target) != expected:
            raise RuntimeError(f"source changed while snapshotting: {relative}; use a new output")
    upstream_files = ("pysot/models/backbone/newalexnet.py", "pysot/models/utile/utile.py",
                      "pysot/models/utile/tran.py", "pysot/models/loss.py", "pysot/datasets/anchortarget.py",
                      "pysot/core/config.py", "experiments/config.yaml", "tools/train.py")
    upstream_hashes = {}
    for relative in upstream_files:
        source = ROOT / "third_party/HiFT" / relative
        target = sources / "third_party/HiFT" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        upstream_hashes[relative] = sha256(target)
    with (output / "dirty.diff").open("xb") as stream:
        stream.write(git_output(ROOT, "diff", "HEAD"))
    checkpoint = ROOT / "artifacts/checkpoints/general_model.pth"
    manifest = {"schema": "hift-b2-synthetic-profile-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "identity": "B2 corrected-training feasibility; random initialization; NOT B0/B1 accuracy",
                "source": identity, "upstream_files_sha256": upstream_hashes,
                "checkpoint": {"initialization": "random", "official_file_exists": checkpoint.is_file(),
                               "official_file_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
                               "loaded": False, "reason": "layout repair changes learned computation; synthetic feasibility only"},
                "data": {"kind": "synthetic uniform [0,255] resident crops", "split": "none", "template": 127,
                         "search": 287, "bbox": [95, 95, 223, 223], "negative_policy": "dense background; zero regression",
                         "timed_negative_fraction": "B1 alternating, B2/4 50%; not upstream 5% distribution"},
                "settings": {"device": str(device), "seed": seed, "dtype": "float32", "tf32": False,
                             "deterministic_algorithms": True, "cudnn_benchmark": False,
                             "threads": threads, "warmup": warmup, "repeats": repeats, "batches": list(batches),
                             "gpu_token": gpu_token, "sgd": {"backbone_lr": 0.0005, "grader_lr": 0.005,
                             "momentum": 0.9, "weight_decay": 0.0001, "clip_norm": 10},
                             "loss": "2.4*masked upstream IoU + balanced NLL + mean BCEWithLogits",
                             "backbone": "layers1-2 frozen + BN eval; layers3-5 trainable + BN train",
                             "dropout": "train-mode upstream transformer 0.1; Cattention dropout 0.5",
                             "mask_order": "select raw loc before inverse-tanh and ignored cls1 before log_softmax",
                             "decode": "conditional +/-0.99 saturation; log1p mathematically equivalent, not bit-exact upstream log(1+x)",
                             "argv": sys.argv},
                "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                                "cpu": platform.processor(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
                                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                                "torch_config": torch.__config__.show()},
                "results": []}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        manifest["environment"]["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                                            "capability": [props.major, props.minor]}
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            manifest["environment"]["nvidia_smi"] = {"returncode": smi.returncode,
                                                      "stdout": smi.stdout.strip(), "stderr": smi.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            manifest["environment"]["nvidia_smi"] = {"unavailable": repr(exc)}
    write_json(output / "manifest.start.json", manifest)
    for batch in batches:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        started = time.perf_counter()
        try:
            result = profile_batch(batch, device, seed=seed, warmup=warmup, repeats=repeats)
        except torch.OutOfMemoryError as exc:
            result = {"status": "oom", "batch_size": batch, "error": str(exc), "memory": memory(device),
                      "note": "no throughput inferred for failed batch"}
        except Exception as exc:
            result = {"status": "error", "batch_size": batch, "error": repr(exc), "memory": memory(device)}
        result["wall_seconds_including_setup_probes"] = time.perf_counter()-started
        write_json(output / f"batch-{batch}.json", result)
        manifest["results"].append(result)
        print(json.dumps({"batch": batch, "status": result["status"],
                          "pairs_per_second": result.get("synthetic_pairs_per_second"),
                          "error": result.get("error")}), flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_files_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest
