"""Source-derived, explicitly proposed B2 recipe. This module authorizes no training."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from hift_research import UPSTREAM_COMMIT
from hift_research.provenance import sha256

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "third_party/HiFT/tools/train.py", "third_party/HiFT/pysot/utils/lr_scheduler.py",
    "third_party/HiFT/pysot/core/config.py", "third_party/HiFT/experiments/config.yaml",
    "third_party/HiFT/pysot/utils/model_load.py",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_settings(root=ROOT):
    """Read literal defaults from the actual source, then recursively merge released YAML."""
    root = Path(root)
    defaults = {}
    tree = ast.parse((root / SOURCE_FILES[2]).read_text(encoding="utf-8"))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
        for target in statement.targets:
            parts = []
            while isinstance(target, ast.Attribute):
                parts.insert(0, target.attr)
                target = target.value
            if not isinstance(target, ast.Name) or target.id != "__C" or not parts:
                continue
            destination = defaults
            for part in parts[:-1]:
                destination = destination.setdefault(part, {})
            destination[parts[-1]] = value
    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict):
                merge(destination.setdefault(key, {}), value)
            else:
                destination[key] = value
    merge(defaults, yaml.safe_load((root / SOURCE_FILES[3]).read_text(encoding="utf-8")))
    return json.loads(json.dumps(defaults))


def source_lr_table(settings=None):
    """Exact released step-warmup/log table + optimizer-base normalization (70 epochs)."""
    settings = source_settings() if settings is None else settings
    train = settings["TRAIN"]
    warm = train["LR_WARMUP"]
    normal = train["LR"]
    if not warm["WARMUP"] or warm["TYPE"] != "step" or normal["TYPE"] != "log":
        raise ValueError("this audited recipe requires pinned step-warmup + log schedule")
    count = train["EPOCH"]
    warm_count = warm["EPOCH"]
    kw = warm["KWARGS"]
    multiplier = math.pow(kw["end_lr"]/kw["start_lr"], 1.0/(warm_count//kw["step"]))
    warm_values = kw["start_lr"] * multiplier ** (np.arange(warm_count)//kw["step"])
    kw_normal = normal["KWARGS"]
    normal_values = np.logspace(math.log10(kw_normal["start_lr"]),
                               math.log10(kw_normal["end_lr"]), count-warm_count)
    spaces = np.concatenate((warm_values, normal_values))
    return (spaces * train["BASE_LR"] / normal_values[0]).tolist()


def make_proposed_recipe(*, training_manifest=None, root=ROOT):
    root = Path(root)
    settings = source_settings(root)
    train, backbone = settings["TRAIN"], settings["BACKBONE"]
    asset = root / "third_party/HiFT/pretrained_models" / backbone["PRETRAINED"]
    recipe = {
        "schema": "hift-b2-wave3-proposed-v1", "identity": "B2 proposed corrected training; not baseline accuracy",
        "status": "provisional_fail_closed", "launch_allowed": False,
        "authorization": "CPU synthetic lifecycle only; real-data SGD and GPU forbidden in Wave3 task9",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_files_sha256": {relative: sha256(root/relative) for relative in SOURCE_FILES},
        "source_merged_settings": settings,
        "optimizer": {"kind": "SGD", "base_lr": train["BASE_LR"], "backbone_lr_ratio": backbone["LAYERS_LR"],
                      "momentum": train["MOMENTUM"], "weight_decay": train["WEIGHT_DECAY"],
                      "clip_norm": train["GRAD_CLIP"], "preserve_momentum_at_unfreeze": True,
                      "group_membership": "all backbone params + all grader params, stable across epochs"},
        "schedule": {"unit": "completed data epochs, never optimizer updates", "epochs": train["EPOCH"],
                     "grader_lr_by_epoch": source_lr_table(settings), "backbone_unfreeze_epoch": backbone["TRAIN_EPOCH"],
                     "train_layers": backbone["TRAIN_LAYERS"], "freeze_layers1_2_bn_always": True},
        "batch": {"source_global": train["BATCH_SIZE"], "source_num_gpu": train["NUM_GPU"],
                  "proposed_processes": 1, "proposed_microbatch": 4, "accumulation_microbatches": 55,
                  "effective_batch": 220, "final_partial": "normalize actual counts; step once; never discard",
                  "normalization": "global per-class NLL counts, BCE element count, IoU weight sum +1e-6",
                  "batchnorm": "microbatch statistics; NOT equivalent to global batch220",
                  "dropout": "upstream train mode; no microbatch equivalence claim"},
        "input_contract": {"candidate_negative_probability": settings["DATASET"]["NEG"],
                           "target_weight_policy": "retain upstream64 boundary behavior; audit frequencies, no retuning",
                           "identity": "canonical dataset/video/track; identity collisions rejected, attempts counted separately",
                           "training_manifest": None, "training_manifest_sha256": None, "validated_real_data": False},
        "initialization": {"backbone_source_file": str(asset.relative_to(root)).replace("\\", "/"),
                           "backbone_sha256": sha256(asset) if asset.is_file() else None,
                           "required_load": "weights_only strict normalized keys; ONLY five absent legacy BN num_batches_tracked counters -> zero-int64",
                           "grader": "random seed123456 (source CLI default), gamma starts0",
                           "official_released_tracker_checkpoint": "not a training initialization substitute"},
        "deviations": [
            "fixed optimizer group membership and preserve grader momentum at epoch10 (upstream rebuild resets it)",
            "layers1-2 BN always eval, including fresh resume>=10 (source mode depends on entry path)",
            "one-process accumulated batch220 instead of source two-GPU DataParallel; BN/dropout/reductions differ",
            "global class/mask denominators instead of averaging independently normalized microbatch/device losses",
            "final partial window normalized by actual observations rather than discarded/truncated epoch tail",
            "strict model keys, optimizer/scheduler/RNG/sampler/pending-gradient checkpoint resume",
            "explicit B2 layout/target/classification/nonlinearity-mask corrections; frozen source graph adapters",
            "real recipe cannot launch until legal manifest binding, asset validation, budget and separate authorization",
        ],
        "blockers": ["no validated canonical real training manifest bound", "no real-data SGD authorization",
                     "no full pipeline/accumulation runtime measurement", "backbone requires approved five-counter-only compatibility audit (reported separately)"],
    }
    if training_manifest is not None:
        bind_training_manifest(recipe, training_manifest, root=root)
    return recipe


def bind_training_manifest(recipe, manifest_path, *, root=ROOT):
    """Bind authentic TRAIN provenance for preflight ONLY, never production authority.

    An evaluation registry or synthetic manifest cannot satisfy this link. Canonical
    COCO annotation-instance IDs are not global physical IDs or a legal opinion.
    Distribution/leakage/recipe acceptance remain independent gates.
    """
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path(root) / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "hift-training-data-v1" or manifest.get("status") != "frozen_preflight_only":
        raise ValueError("canonical frozen preflight training manifest required; provisional gate remains closed")
    try:
        from hift_research.data_contract.schema import validate_training_manifest
    except ImportError as exc:
        raise ValueError("canonical manifest validator unavailable; provisional gate remains closed") from exc
    manifest = validate_training_manifest(path, verify_files=True)
    if (manifest.get("dataset") != "COCO2017" or manifest.get("split") != "train2017"
            or not manifest.get("instances") or manifest.get("policy", {}).get("candidate_negative_probability") != 0.05
            or manifest.get("validation", {}).get("production_training_authorized") is not False):
        raise ValueError("canonical train2017/preflight-only contract mismatch")
    recipe["input_contract"].update({"training_manifest": str(path), "training_manifest_sha256": sha256(path),
                                     "validated_manifest": True, "validated_real_data": False,
                                     "preflight_only": True, "reservations_sha256": manifest.get("reservations_sha256"),
                                     "source_code_hashes": manifest.get("source_code_hashes"),
                                     "negative_alias_policy": manifest["policy"].get("alias_unknown")})
    recipe["status"] = "manifest_linked_provisional_not_authorized"
    recipe["blockers"] = [b for b in recipe["blockers"] if b != "no validated canonical real training manifest bound"]
    recipe["blockers"].append("canonical manifest validates preflight only, not accepted distribution or real training readiness")
    return recipe


def load_legacy_backbone_strict(backbone, checkpoint_path):
    """Approved minimal compatibility: exactly five absent legacy BN counters -> int64 zero.

    Never remove unexpected keys, silently ignore learned/running state, or load
    partial weights. Fixed BN momentum .1 uses EMA independent of these counters;
    cumulative-average BatchNorm (momentum=None) is explicitly unsupported.
    """
    import torch
    path = Path(checkpoint_path)
    checkpoint_hash = sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weights = payload.get("state_dict", payload)
    expected = backbone.state_dict()
    counters = {f"layer{i}.1.num_batches_tracked" for i in range(1, 6)}
    actual_counters = {name for name in expected if name.endswith("num_batches_tracked")}
    if actual_counters != counters or any(module.momentum != 0.1 for module in backbone.modules()
                                          if isinstance(module, torch.nn.BatchNorm2d)):
        raise ValueError("only pinned five fixed-momentum BatchNorm layers supported")
    missing, unexpected = set(expected)-set(weights), set(weights)-set(expected)
    if unexpected or missing not in (set(), counters):
        raise ValueError("legacy compatibility permits only exactly five absent BN counters")
    normalized = {}
    original_hashes = {}
    for name, value in weights.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
            raise ValueError("backbone checkpoint shape/dtype/finite mismatch")
        normalized[name] = value.detach().clone()
        original_hashes[name] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    for name in missing:
        if expected[name].dtype != torch.int64 or expected[name].numel() != 1:
            raise ValueError("legacy BN counter must be scalar int64")
        normalized[name] = torch.zeros_like(expected[name])
    backbone.load_state_dict(normalized, strict=True)
    loaded = backbone.state_dict()
    preserved = all(torch.equal(loaded[name].cpu(), value.cpu()) for name, value in weights.items())
    zeros = all(loaded[name].dtype == torch.int64 and int(loaded[name]) == 0 for name in missing)
    if not preserved or not zeros or sha256(path) != checkpoint_hash:
        raise ValueError("backbone checkpoint preservation/counter/hash validation failed")
    return {"path": str(path), "sha256": checkpoint_hash, "weights_only": True,
            "raw_missing_keys": sorted(missing), "raw_unexpected_keys": sorted(unexpected),
            "strict_raw_keys_match": not missing and not unexpected, "strict_normalized_load": True,
            "normalization": "only five scalar int64 BN counters initialized to zero" if missing else "none",
            "inserted_zero_counters": sorted(missing), "all_original_tensors_bitexact": preserved,
            "original_tensor_sha256": original_hashes, "checkpoint_file_unchanged": True,
            "bn_momentum": 0.1, "cumulative_bn_supported": False, "loaded_for_optimization": False}


def validate_proposed_config(config, *, root=ROOT):
    """Machine-check executable numeric choices and an optional immutable preflight binding."""
    recipe = make_proposed_recipe(root=root)
    if (config.get("launch_allowed") is not False or config.get("real_data_sgd_authorized") is not False
            or config.get("gpu_authorized_for_lifecycle") is not False or config.get("validated_real_data") is not False
            or config.get("source_derived_recipe_sha256") != canonical_hash(recipe)):
        raise ValueError("proposed config authorization/source recipe drift")
    source = recipe["source_merged_settings"]
    train, backbone = source["TRAIN"], source["BACKBONE"]
    expected = {"epochs": train["EPOCH"], "source_pairs_per_epoch": source["DATASET"]["VIDEOS_PER_EPOCH"],
                "source_global_batch": train["BATCH_SIZE"], "source_num_gpu": train["NUM_GPU"],
                "source_base_lr": train["BASE_LR"], "backbone_lr_ratio": backbone["LAYERS_LR"],
                "momentum": train["MOMENTUM"], "weight_decay": train["WEIGHT_DECAY"], "clip_norm": train["GRAD_CLIP"],
                "backbone_unfreeze_epoch": backbone["TRAIN_EPOCH"], "train_layers": backbone["TRAIN_LAYERS"],
                "candidate_negative_probability": source["DATASET"]["NEG"],
                "source_backbone_initialization_sha256": recipe["initialization"]["backbone_sha256"]}
    actual = config.get("proposed_source_preserving_settings", {})
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("proposed config numeric settings drift from pinned source")
    manifest_path = config.get("real_training_manifest")
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = Path(root)/path
        if sha256(path) != config.get("real_training_manifest_sha256"):
            raise ValueError("frozen config training manifest hash mismatch")
        bind_training_manifest(recipe, path, root=root)
        expected_hashes = {str(Path(name).relative_to(root)).replace("\\", "/"): value
                           for name, value in recipe["input_contract"]["source_code_hashes"].items()}
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (config.get("bound_recipe_sha256") != canonical_hash(recipe)
                or config.get("manifest_source_code_hashes") != expected_hashes
                or config.get("reservations_sha256") != recipe["input_contract"]["reservations_sha256"]
                or config.get("manifest_policy_sha256") != manifest["policy_sha256"]
                or canonical_hash(manifest["policy"]) != manifest["policy_sha256"]):
            raise ValueError("frozen config source/recipe/policy/reservations mapping mismatch")
        for relative, expected_hash in expected_hashes.items():
            if sha256(Path(root)/relative) != expected_hash:
                raise ValueError(f"frozen canonical input source hash mismatch: {relative}")
    elif config.get("real_training_manifest_sha256") is not None:
        raise ValueError("manifest hash has no bound training manifest path")
    return recipe


def validate_recipe_binding(recipe, *, root=ROOT):
    """Check immutable config data identity. Never turn this into real-training permission."""
    if recipe.get("launch_allowed") is not False:
        raise ValueError("this proposed recipe cannot authorize a real-data launch")
    contract = recipe["input_contract"]
    if not contract.get("validated_manifest") or not contract.get("training_manifest"):
        raise ValueError("provisional: canonical real training data binding absent")
    path = Path(contract["training_manifest"])
    if not path.is_absolute():
        path = Path(root)/path
    if sha256(path) != contract["training_manifest_sha256"]:
        raise ValueError("frozen training manifest hash mismatch")
    check = copy.deepcopy(recipe)
    bind_training_manifest(check, path, root=root)
    return True
