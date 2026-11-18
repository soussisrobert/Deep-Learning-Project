"""Future explicitly authorized head-only pilot; shared serial production controller.

No real SGD permission exists in Wave4. Missing exact future human grant fails
before CUDA/model construction. Boundary replay is the only supported fallback.
"""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hift_research.provenance import sha256, source_identity
from .graph import B2TrainingGraph
from .lifecycle import actual_heads, apply_epoch_modes, window_denominators, window_loss
from .recipe import ROOT, load_legacy_backbone_strict
from .wave4_lifecycle import device_tree, seed_device
from .wave4_loader import indexed_loader
from .wave4_layout import canonicalize_model_inputs
from .wave4_recipes import validate_frozen_pilot, validate_temporal_binding


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def future_authorization(config_path, authorization_file=None, *, ack_real_sgd=False):
    """Fail before ANY CUDA/model initialization without a separate post-Wave4 human grant."""
    if not ack_real_sgd or authorization_file is None:
        raise PermissionError("NOT AUTHORIZED: separate post-Wave4 human authorization file AND --ack-real-sgd required")
    config_path, authorization_file = Path(config_path).resolve(), Path(authorization_file).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_pilot(config)
    grant = json.loads(authorization_file.read_text(encoding="utf-8"))
    expected = {"schema": "hift-explicit-future-pilot-authorization-v1", "human_approved": True,
                "separate_post_wave4_request": True, "allow_real_sgd": True, "config_sha256": sha256(config_path),
                "training_manifest_sha256": config["data"]["manifest_sha256"],
                "training_policy_sha256": config["data"]["train_policy_sha256"],
                "development_policy_sha256": config["data"]["development_policy_sha256"],
                "max_effective_updates": 100, "max_training_attempts": 1600, "device": "cuda:0"}
    if any(grant.get(key) != value for key, value in expected.items()) or not grant.get("human_request_reference") or not grant.get("exclusive_gpu_token"):
        raise PermissionError("future human grant scope/config/data/budget mismatch")
    try:
        expires = datetime.fromisoformat(grant["expires_utc"])
        valid_time = expires.tzinfo is not None and expires > datetime.now(timezone.utc)
    except (ValueError, KeyError, TypeError):
        valid_time = False
    if not valid_time:
        raise PermissionError("future authorization absent/expired/not timezone qualified")
    if any(config["data"][key] is None for key in ("manifest", "manifest_sha256", "train_policy", "train_policy_sha256",
                                                    "development_policy", "development_policy_sha256", "reservations_sha256")):
        raise PermissionError("data/policy/reservations not frozen")
    if not config["development"]["cohort_manifest"] or not config["development"]["cohort_sha256"]:
        raise PermissionError("fixed balanced development cohort not bound")
    if sha256(ROOT/config["development"]["cohort_manifest"]) != config["development"]["cohort_sha256"]:
        raise PermissionError("fixed development cohort hash drift")
    seconds = config["budget"]["max_wall_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise PermissionError("measured full-input-path wall budget absent/unsafe")
    if grant.get("max_wall_seconds") != seconds:
        raise PermissionError("human wall budget differs from frozen full-input budget")
    if config["readiness"]["recommendation"] != "GO_RECOMMENDATION_NOT_AUTHORIZATION":
        raise PermissionError("readiness has not recommended this exact pilot")
    return config, grant


def attempt_batch(dataset, ids, *, kind):
    if not ids:
        raise ValueError("empty attempt group")
    batch = next(iter(indexed_loader(dataset, start=ids[0], batch_size=len(ids), num_workers=0, errors="raise")))
    if batch["attempt_indices"] != ids:
        raise ValueError("indexed serial loader attempt order mismatch")
    values = batch["model_inputs"]
    if values is None:
        values = {"template": torch.empty(0, 3, 127, 127), "search": torch.empty(0, 3, 287, 287),
                  "label_cls1": torch.empty(0, 1, 11, 11), "labelcls2": torch.empty(0, 1, 11, 11),
                  "labelxff": torch.empty(0, 4, 11, 11), "weightxff": torch.empty(0, 1, 11, 11)}
    values = canonicalize_model_inputs(values)
    return {"data_kind": kind, "sample_ids": ids, "template": values["template"], "search": values["search"],
            "targets": {key: values[key] for key in ("label_cls1", "labelcls2", "labelxff", "weightxff")},
            "attempt_records": batch["attempt_records"], "accepted_records": batch["accepted_records"],
            "rejected_records": batch["rejected_records"], "replacement_count": batch["replacement_count"]}


def buffered_attempt_window(dataset, ids, physical=4, *, kind="authorized_future_real"):
    batches = [attempt_batch(dataset, ids[start:start+physical], kind=kind) for start in range(0, len(ids), physical)]
    if any(batch["replacement_count"] for batch in batches):
        raise ValueError("replacement forbidden")
    return batches, window_denominators(batches)


def paired_development_loss(model, dataset, device, *, expected_attempts=128):
    """Future eval only: fixed pairs, eval BN/dropout, no backward/SGD; not tracking AUC."""
    batches, denominators = buffered_attempt_window(dataset, list(range(expected_attempts)), kind="real_development_forward_only")
    records = [r for b in batches for r in b["accepted_records"]]
    positive, negative = sum(not r["candidate_negative"] for r in records), sum(bool(r["candidate_negative"]) for r in records)
    if positive != 64 or negative != 64 or any(b["rejected_records"] for b in batches):
        raise ValueError("predeclared 64+64 dev cohort incomplete; do not refill/retune")
    model.eval()
    loss = 0.0
    with torch.inference_mode():
        for cpu_batch in batches:
            batch = device_tree(cpu_batch, device)
            loss += float(window_loss(actual_heads(model, batch), batch["targets"], denominators))
    apply_epoch_modes(model, 0)
    if not math.isfinite(loss):
        raise ValueError("nonfinite fixed development loss")
    return {"total_loss": loss, "positive_pairs": positive, "negative_pairs": negative,
            "attempts": expected_attempts, "tracking_auc": None, "selection": "fixed pair loss only"}


def launch_future_pilot(config_path, *, authorization_file=None, ack_real_sgd=False, output=None, resume=None, resume_sha256=None):
    # FIRST operation: current Wave4 must never create a grant or invoke this successfully.
    config, grant = future_authorization(config_path, authorization_file, ack_real_sgd=ack_real_sgd)
    if resume and (not resume_sha256 or sha256(resume) != resume_sha256):
        raise ValueError("explicit immutable resume checkpoint SHA256 required")
    dataset = validate_temporal_binding(config)
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    from .wave4_pilot_engine import SerialPilotController
    dev = TemporalCandidateDataset(ROOT/config["data"]["manifest"], ROOT/config["data"]["development_policy"],
                                   split="development", seed=20261024, length=128)
    if output is None:
        raise ValueError("new exclusive output directory required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    device = seed_device(config["seed"], config["device"], gpu_token=grant["exclusive_gpu_token"])
    model = B2TrainingGraph()
    initialization = load_legacy_backbone_strict(model.backbone, ROOT/config["source_backbone_path"])
    if initialization["sha256"] != config["source_backbone_sha256"]:
        raise ValueError("source backbone identity drift")
    identity = {"kind": "authorized_future_real", "manifest_sha256": dataset.manifest_sha256,
                "policy_sha256": dataset.policy_sha256, "reservations_sha256": dataset.reservations_sha256}
    controller = SerialPilotController(model, dataset, config_identity={"config_sha256": sha256(config_path),
                         "runner_sha256": sha256(ROOT/"scripts/run_b2_pilot.py")}, data_identity=identity,
                         device=device, gpu_token=grant["exclusive_gpu_token"], synthetic=False,
                         max_updates=100, max_attempts=1600, config_path=config_path,
                         authorization_file=authorization_file, ack_real_sgd=ack_real_sgd)
    dev_results, above, previous_wall = [], 0, 0.0
    if resume:
        controller.restore_boundary(resume, expected_sha256=resume_sha256)
        history = controller.auxiliary_state
        dev_results, above, previous_wall = history["development"], history["above_limit_count"], history["wall_elapsed"]
    deadline = start+config["budget"]["max_wall_seconds"]-previous_wall
    manifest = {"schema": "hift-explicit-future-pilot-run-v1", "authorization_sha256": sha256(authorization_file),
                "config_sha256": sha256(config_path), "source": source_identity(ROOT), "data": config["data"],
                "seed": config["seed"], "device": str(device), "started_utc": datetime.now(timezone.utc).isoformat(),
                "real_sgd": True, "initialization": initialization, "resume": str(resume) if resume else None,
                "resume_policy": "explicit atomic boundary replay only; no automatic retry; repeated I/O is reported, not a replacement"}
    exclusive_json(output/"manifest.start.json", manifest)
    checkpoints = []
    def checkpoint_and_development():
        nonlocal above
        result = paired_development_loss(model, dev, device)
        result["update"] = controller.update
        dev_results.append(result)
        baseline = max(dev_results[0]["total_loss"], 1e-6)
        above = above+1 if result["total_loss"] > 3*baseline else 0
        controller.auxiliary_state = {"development": dev_results, "above_limit_count": above,
                                      "wall_elapsed": previous_wall+time.perf_counter()-start}
        checkpoints.append(controller.save_boundary(output/f"update-{controller.update:04d}.pt"))
        if above >= 2:
            raise ValueError("two consecutive fixed-dev losses exceed3xbaseline")
        if time.perf_counter() > deadline:
            raise TimeoutError("pilot budget exceeded including development/checkpoint cost")
    try:
        if not resume:
            checkpoint_and_development()
        while controller.cursor < 1600 and controller.update < 100:
            if time.perf_counter() > deadline:
                raise TimeoutError("measured future wall budget exceeded")
            result = controller.step(deadline_monotonic=deadline)
            memory = result["measurement"]["memory"]
            if memory["reserved"]/torch.cuda.get_device_properties(device).total_memory > 0.8:
                raise MemoryError("future pilot CUDA reserved fraction exceeded0.8")
            if result["semantic"]["stepped"] and controller.update in (25, 50, 100):
                checkpoint_and_development()
        if controller.update != 100:
            raise ValueError("1600attempt budget exhausted before100updates; no refill")
        if dev_results[-1]["total_loss"] > 3*max(dev_results[0]["total_loss"], 1e-6):
            raise ValueError("fixed-last primary development criterion failed")
        manifest.update({"result": "completed bounded head-only reliability pilot; not tracking accuracy", "updates": controller.update,
                         "training_unique_committed_attempts": controller.cursor, "trace": controller.semantic_trace,
                         "measurements": controller.measurements, "development": dev_results, "checkpoints": checkpoints,
                         "primary_checkpoint_update": 100,
                         "secondary_best_dev_update": min(dev_results, key=lambda r:(r["total_loss"],r["update"]))["update"],
                         "wall_seconds": previous_wall+time.perf_counter()-start})
        exclusive_json(output/"manifest.json", manifest)
    except Exception as exc:
        exclusive_json(output/"failure.json", {"error": f"{type(exc).__name__}:{exc}", "updates": controller.update,
                                               "committed_attempt_position": controller.cursor, "uncommitted_window": controller.in_window,
                                               "trace": controller.semantic_trace, "wall_seconds_this_invocation": time.perf_counter()-start,
                                               "no_automatic_retry": True, "boundary_checkpoint_only": True})
        raise
    return manifest
