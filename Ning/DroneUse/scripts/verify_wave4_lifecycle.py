"""Bounded synthetic device replay via indexed Dataset/collate/H2D; never real SGD."""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hift_research.provenance import sha256, source_identity
from hift_research.training.graph import B2TrainingGraph, gradient_audit
from hift_research.training.lifecycle import SyntheticSampler
from hift_research.training.recipe import ROOT, canonical_hash
from hift_research.training.wave4_loader import SyntheticCandidateDataset, SyntheticIndexedFactory, dataset_identity
from hift_research.training.wave4_lifecycle import Wave4SyntheticLifecycle, cpu_tree, device_rng, seed_device
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
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def run(output, *, device="cpu", gpu_token=None, config_path=ROOT/"configs/B2-Pilot-W4.json", physical=4, effective=16, updates=2):
    device = torch.device(device)
    if device.type == "cuda" and not gpu_token:
        raise ValueError("lead GPU token required; no CUDA operation permitted without it")
    epochs = (0, 10)
    total_updates = updates*2*len(epochs)
    if updates < 1 or updates > 12 or total_updates > 12 or effective % physical or physical < 1:
        raise ValueError("predeclared synthetic budget<=12TOTALupdates, integral accumulation")
    config = json.loads(Path(config_path).read_text())
    validate_frozen_pilot(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = source_identity(ROOT)
    identity = {"pilot_config_sha256": sha256(config_path), "driver_sha256": sha256(__file__),
                "synthetic_physical": physical, "synthetic_effective": effective, "updates_per_path": updates}
    manifest = {"schema": "hift-wave4-synthetic-lifecycle-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source": source, "config_identity": identity, "seed": 20261023, "device": str(device),
                "gpu_token": gpu_token, "total_synthetic_updates": total_updates, "epochs_tested": epochs,
                "authorization": "bounded synthetic only; ALL random weights; no real data or pretrained weights enter SGD",
                "claims": "same-partition exact replay, NOT physical220 equivalence or real-data training readiness",
                "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                                "platform": platform.platform(), "cpu": platform.processor(), "threads": 1}}
    write_json(output/"manifest.start.json", manifest)
    write_json(output/"pilot-config.snapshot.json", config)
    for relative, expected in source["source_files_sha256"].items():
        if relative.startswith(("src/", "scripts/", "tests/", "configs/", "requirements/")):
            target = output/"sources"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((ROOT/relative).read_bytes())
            if sha256(target) != expected:
                raise ValueError(f"source changed while snapshotting:{relative}")
    results = []
    for epoch in epochs:
        seed_device(20261023, device, gpu_token=gpu_token)
        dataset = SyntheticCandidateDataset(length=updates*effective)
        factory = SyntheticIndexedFactory(dataset)
        def engine():
            return Wave4SyntheticLifecycle(B2TrainingGraph(), SyntheticSampler(range(len(dataset)), epoch=epoch),
                                           batch_factory=factory.batch, config_identity=identity,
                                           data_identity={"kind": "synthetic", "sha256": dataset.manifest_sha256, **dataset_identity(dataset)},
                                           microbatch=physical, accumulation=effective//physical, epoch=epoch,
                                           device=device, gpu_token=gpu_token, max_updates=updates)
        full = engine()
        full.microstep()
        first_audit = gradient_audit(full.model)
        full.run_to_end()
        full_rng = device_rng(device)
        seed_device(20261023, device, gpu_token=gpu_token)
        part = engine()
        cut = max(1, effective//physical//2)
        for _ in range(cut):
            part.microstep()
        checkpoint = output/f"epoch{epoch}-midaccum.pt"
        part.save(checkpoint)
        restored = engine().restore(checkpoint)
        restored.run_to_end()
        comparisons = {
            "model_parameters_bn_buffers_bitexact": equal_tree(full.model.state_dict(), restored.model.state_dict()),
            "optimizer_momentum_bitexact": equal_tree(full.optimizer.state_dict(), restored.optimizer.state_dict()),
            "scheduler_epoch_lr_bitexact": equal_tree(full.scheduler.state_dict(), restored.scheduler.state_dict()),
            "trace_order_loss_bitexact": equal_tree(full.trace, restored.trace),
            "sampler_consumed_position_bitexact": equal_tree(full.sampler.state_dict(), restored.sampler.state_dict()),
            "pending_gradients_bitexact": equal_tree({n:p.grad for n,p in full.model.named_parameters()}, {n:p.grad for n,p in restored.model.named_parameters()}),
            "python_numpy_torch_cpu_cuda_rng_bitexact": equal_tree(full_rng, device_rng(device)),
            "pending_cleared": restored.pending is None and full.pending is None,
            "exact_update_budget": restored.update == full.update == updates,
        }
        if not all(comparisons.values()):
            raise AssertionError(f"unexplained device resume drift epoch{epoch}:{comparisons}")
        results.append({"epoch": epoch, "comparisons": comparisons, "trace": restored.trace,
                        "physical": physical, "nominal_effective": effective, "updates_each_path": updates,
                        "checkpoint_sha256": sha256(checkpoint), "first_gradient_audit": first_audit,
                        "final_gradient_audit": gradient_audit(restored.model), "device_identity": restored.device_identity,
                        "bn_modes": {n:m.training for n,m in restored.model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)},
                        "gamma": float(restored.model.grader.transformer.encoder.layers[0].cross_attn.gamma.detach())})
        del full, part, restored
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        manifest["environment"]["gpu"] = torch.cuda.get_device_name(device)
        manifest["environment"]["cuda"] = torch.version.cuda
    write_json(output/"verification.json", {"cases": results})
    manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter()-started,
                     "result": "PASS bounded synthetic device lifecycle", "cases_passed": len(results),
                     "production_num_workers": 0, "prefetch": False,
                     "limitations": "in-process fresh-model reconstruction, no asynchronous worker/prefetch exact resume claim; no real SGD",
                     "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}})
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"output": str(output), "result": manifest["result"], "updates_total": total_updates,
                      "wall_seconds": manifest["wall_seconds"], "cases": [r["comparisons"] for r in results]}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-token")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/B2-Pilot-W4.json")
    parser.add_argument("--physical", type=int, default=4)
    parser.add_argument("--effective", type=int, default=16)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    run(args.output, device=args.device, gpu_token=args.gpu_token, config_path=args.config,
        physical=args.physical, effective=args.effective, updates=args.updates)
