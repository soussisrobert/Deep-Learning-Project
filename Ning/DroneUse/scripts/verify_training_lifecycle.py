"""Immutable CPU-only synthetic lifecycle evidence; never optimizes authentic data."""
import argparse
import copy
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hift_research.provenance import sha256, source_identity
from hift_research.training.graph import B2TrainingGraph, gradient_audit, synthetic_batch
from hift_research.training.lifecycle import (
    SyntheticLifecycle, SyntheticSampler, apply_epoch_modes, capture_rng, seed_cpu,
)
from hift_research.training.recipe import (
    ROOT, canonical_hash, load_legacy_backbone_strict, make_proposed_recipe, validate_proposed_config,
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def synthetic_factory(ids):
    template, search, targets = synthetic_batch(len(ids), negative=[bool(i % 3 == 0) for i in ids])
    # All three RNG streams influence synthetic data. NO real image loader is used.
    factor = 0.9 + random.random()*0.05 + float(np.random.random())*0.05
    return {"data_kind": "synthetic", "sample_ids": ids, "template": template*factor,
            "search": search, "targets": targets}


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def run(output, *, training_manifest=None, recipe_config=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    seed = 20261023
    seed_cpu(seed)
    recipe = make_proposed_recipe(training_manifest=training_manifest)
    config = json.loads(Path(recipe_config).read_text()) if recipe_config else None
    if config is not None:
        validated = validate_proposed_config(config)
        if canonical_hash(validated) != canonical_hash(recipe):
            raise ValueError("config/command canonical training-manifest binding differs")
    source = source_identity(ROOT)
    manifest = {"schema": "hift-wave3-lifecycle-verification-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source": source, "recipe_sha256": canonical_hash(recipe),
                "config_file_sha256": sha256(recipe_config) if recipe_config else None,
                "data_identity": {"kind": "synthetic", "sha256": canonical_hash([1, 2, 4]),
                                  "description": "uniform synthetic crops; no authentic data SGD"},
                "seed": seed, "cpu_only": True, "gpu_used": False,
                "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                "platform": platform.platform(), "cpu": platform.processor(), "threads": torch.get_num_threads()},
                "authorization": "bounded CPU synthetic optimization only; real B2 and GPU forbidden",
                "claims": "lifecycle correctness only, not real-data readiness, convergence or accuracy"}
    write_json(output/"manifest.start.json", manifest)
    write_json(output/"recipe.snapshot.json", recipe)
    # Copy every source that may execute; exact file bytes, including untracked code.
    for relative, expected in source["source_files_sha256"].items():
        if not relative.startswith(("src/", "scripts/", "tests/", "configs/", "requirements/")):
            continue
        target = output/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT/relative).read_bytes()
        with target.open("xb") as stream:
            stream.write(content)
        if sha256(target) != expected:
            raise ValueError(f"source changed while snapshotting {relative}")
    for relative in recipe["source_files_sha256"]:
        target = output/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT/relative).read_bytes())
    def engine():
        return SyntheticLifecycle(B2TrainingGraph(), SyntheticSampler([1, 2, 4], epoch=10),
                                  batch_factory=synthetic_factory, epoch=10, accumulation=2,
                                  config_identity={"recipe_sha256": canonical_hash(recipe), "factory": "synthetic_factory-v1"},
                                  data_identity=manifest["data_identity"])
    full = engine()
    gamma_initial = full.model.grader.transformer.encoder.layers[0].cross_attn.gamma.item()
    full.microstep()
    first_audit = gradient_audit(full.model)
    full.run_to_end()
    full_rng = capture_rng()
    seed_cpu(seed)
    interrupted = engine()
    interrupted.microstep()
    checkpoint = output/"synthetic-mid-accumulation.pt"
    interrupted.save(checkpoint)
    restored = engine().restore(checkpoint)
    restored.run_to_end()
    comparisons = {
        "sample_order_loss_trace_bitexact": equal_tree(full.trace, restored.trace),
        "model_parameters_buffers_bitexact": equal_tree(full.model.state_dict(), restored.model.state_dict()),
        "optimizer_momentum_bitexact": equal_tree(full.optimizer.state_dict(), restored.optimizer.state_dict()),
        "scheduler_epoch_lr_bitexact": equal_tree(full.scheduler.state_dict(), restored.scheduler.state_dict()),
        "sampler_bitexact": equal_tree(full.sampler.state_dict(), restored.sampler.state_dict()),
        "pending_cleared": full.pending is None and restored.pending is None,
        "gradients_bitexact": equal_tree({n: p.grad for n, p in full.model.named_parameters()},
                                         {n: p.grad for n, p in restored.model.named_parameters()}),
        "python_numpy_torch_rng_bitexact": equal_tree(full_rng, capture_rng()),
        "final_partial_stepped_once": restored.update == 2 and restored.microsteps == 3,
    }
    if not all(comparisons.values()):
        raise AssertionError(f"unexplained resume drift: {comparisons}")
    state_before = {"lr": [g["lr"] for g in restored.optimizer.param_groups],
                    "all_bn": {name: module.training for name, module in restored.model.named_modules()
                               if isinstance(module, torch.nn.BatchNorm2d)}}
    initialization = {}
    asset = ROOT/recipe["initialization"]["backbone_source_file"]
    if asset.is_file():
        # Separate disposable audit model: no pretrained tensor ever enters synthetic SGD.
        audit_backbone = B2TrainingGraph().backbone
        initialization = load_legacy_backbone_strict(audit_backbone, asset)
        initialization.update({"source_upstream_commit": recipe["upstream_commit"],
                               "source_relative_path": recipe["initialization"]["backbone_source_file"],
                               "source_checkpoint_hash_matches_recipe": initialization["sha256"] == recipe["initialization"]["backbone_sha256"]})
        if not initialization["source_checkpoint_hash_matches_recipe"]:
            raise ValueError("source backbone asset hash drift")
    write_json(output/"verification.json", {"comparisons": comparisons, "trace": restored.trace,
               "first_gradient_audit": first_audit, "final_gradient_audit": gradient_audit(restored.model),
               "initial_gamma": gamma_initial, "final_gamma": restored.model.grader.transformer.encoder.layers[0].cross_attn.gamma.item(),
               "epoch10_state": state_before, "backbone_initialization_audit": initialization})
    manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter()-started,
                     "comparisons": comparisons, "synthetic_lifecycle": "PASS bounded CPU correctness",
                     "G2": "PARTIAL: synthetic lifecycle passes; real data/recipe distribution not accepted",
                     "G5": "PARTIAL: provisional real recipe; no real-data training authorization",
                     "checkpoint_sha256": sha256(checkpoint),
                     "artifact_files_sha256": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}})
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"output": str(output), "G2": manifest["G2"], "G5": manifest["G5"],
                      "comparisons": comparisons, "wall_seconds": manifest["wall_seconds"]}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/wave3-lifecycle")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    parser.add_argument("--training-manifest", type=Path, help="optional binding/validation only; NEVER enters optimizer")
    parser.add_argument("--recipe-config", type=Path)
    args = parser.parse_args()
    run(args.output, training_manifest=args.training_manifest, recipe_config=args.recipe_config)


if __name__ == "__main__":
    main()
