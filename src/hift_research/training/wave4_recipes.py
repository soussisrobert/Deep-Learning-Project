"""Reference and predeclared small PILOT policies; neither authorizes real SGD in Wave4."""
import json
from pathlib import Path

from hift_research.provenance import sha256
from .recipe import ROOT, SOURCE_FILES, canonical_hash, source_lr_table, source_settings


def reference_recipe():
    settings = source_settings()
    return {"schema": "hift-b2-reference-w4-v1", "recipe_kind": "B2-Reference-W4",
            "current_wave_real_sgd_authorized": False, "launch_allowed": False,
            "source_files_sha256": {p: sha256(ROOT/p) for p in SOURCE_FILES},
            "source_settings": settings, "grader_lr_by_source_epoch": source_lr_table(settings),
            "source_physical_global_batch": 220, "source_devices": 2, "source_epochs": 70,
            "source_datasets": ["VID", "COCO", "GOT", "YOUTUBEBB"],
            "dataset_mix_validated": False, "initialization": "strict source AlexNet backbone+approved five-zero-counter compatibility; random corrected grader",
            "corrections": "frozen opt-in B2 graph; stable SGD momentum at epoch10 instead of source rebuild; layer1/2 BN always frozen",
            "normalization": "proposed global class/BCE/IoU denominators, not source per-device normalized loss average",
            "physical_batch_equivalence_claim": False,
            "readiness": "NOT a validated 70-epoch reproduction/training recipe; temporal pilot is separate"}


def pilot_recipe():
    settings = source_settings()
    return {"schema": "hift-b2-pilot-w4-v1", "recipe_kind": "B2-Pilot-W4",
            "current_wave_real_sgd_authorized": False,
            "future_authorization_required": True, "future_authorization_file_created": False,
            "reference_recipe": "configs/B2-Reference-W4.json",
            "source_files_sha256": {p: sha256(ROOT/p) for p in SOURCE_FILES},
            "source_backbone_path": "third_party/HiFT/pretrained_models/alexnet-bn.pth",
            "source_backbone_sha256": sha256(ROOT/"third_party/HiFT/pretrained_models/alexnet-bn.pth"),
            "initialization": "strict source backbone five-counter-only compatibility; corrected grader RANDOM; never official general_model",
            "synthetic_verification_initialization": "ALL weights random; NO pretrained weight optimization",
            "seed": 20261023, "device": "cuda:0", "dtype": "float32", "tf32": False,
            "batch": {"attempts_per_microbatch": 4, "attempt_slots_per_window": 16,
                      "nominal_physical": 4, "nominal_effective": 16,
                      "actual": "accepted within fixed attempt slots; may be smaller; normalize observed counts",
                      "refill": False, "replacement_count": 0, "all_reject_window": "log and skip; no optimizer update"},
            "budget": {"max_effective_updates": 100, "max_attempts": 1600, "max_windows": 100,
                       "max_wall_seconds": None, "wall_budget_status": "must bind measured FULL-input-path budget before any future real launch",
                       "max_cuda_reserved_fraction": 0.8},
            "loader": {"num_workers": 0, "prefetch": False, "pin_memory": False, "persistent_workers": False,
                       "indexed_seed": "seed+candidate_index from frozen temporal contract",
                       "resume": "atomic committed-boundary cursor+RNG/model/optimizer; discard/replay entire interrupted window, no asynchronous prefetch replay claim",
                       "image_fault_policy": "STOP with explicit index; no hidden skip/replacement/retry"},
            "optimizer": {"kind": "SGD", "momentum": settings["TRAIN"]["MOMENTUM"],
                          "weight_decay": settings["TRAIN"]["WEIGHT_DECAY"], "clip_norm": settings["TRAIN"]["GRAD_CLIP"],
                          "stable_groups_preserve_momentum": True, "backbone_ratio": 0.1},
            "clock": {"unit": "source data epoch by attempts, not compressed optimizer clock", "source_attempts_per_epoch": 600000,
                      "source_epoch_at_pilot_start": 0, "pilot_crosses_epoch_boundary": False,
                      "grader_lr": source_lr_table(settings)[0], "backbone_lr": source_lr_table(settings)[0]*0.1,
                      "backbone_trainable": False, "backbone_bn_train": False,
                      "unfreeze_at_source_epoch": 10, "pilot_is_head_only": True,
                      "deviation": "100-update reliability pilot is not 70 epochs and does not compress warmup/unfreeze"},
            "data": {"manifest": None, "manifest_sha256": None, "train_policy": None, "train_policy_sha256": None,
                     "development_policy": None, "development_policy_sha256": None, "reservations_sha256": None,
                     "canonical_identity": "source trajectory FRAGMENT, not global person identity",
                     "train_videos": ["MOT17-02", "MOT17-09", "MOT17-10"], "development_videos": ["MOT17-05"],
                     "train_attempts": 1600, "preflight_attempts": 1024, "candidate_negative_probability": 0.05,
                     "negative": "condition eligible simultaneous distinct fragments with full crop A exclusion; explicit source deviation",
                     "positive": "same video/fragment, nonzero temporal gap1..19", "retries_or_replacements": 0},
            "development": {"attempts": 128, "positive_pairs": 64, "safe_negative_pairs": 64, "sampling": "predeclared controlled balanced cohort, NOT training Bernoulli",
                            "class_schedule": "even candidate indices positive, odd known-safe-negative; no resampling",
                            "seed": 20261024, "split": "development", "augmentations": "all shift/scale/blur/color/flip zero",
                            "cohort_manifest": None, "cohort_sha256": None, "require_accepted_positive_and_safe_negative": True,
                            "metric": "fixed paired development loss, global class/mask denominators; NOT tracking AUC",
                            "checkpoints_at_effective_updates": [0, 25, 50, 100],
                            "primary_selection": "fixed last checkpoint update100; do not select on 15 UAV sequences",
                            "secondary_selection": "lowest finite fixed-dev total loss, earliest update breaks ties"},
            "stop_criteria": {"nonfinite_loss_gradient": "immediate stop", "oom_or_device_error": "immediate stop; never infer throughput",
                              "input_fault_or_identity_hash_drift": "immediate stop with exact attempt index",
                              "two_consecutive_dev_losses_gt_baseline_multiple": 3.0,
                              "dev_baseline_denominator_floor": 1e-6, "clip_norm": 10.0,
                              "attempt_budget_exhausted_before_100_updates": "STOP and mark completion FAIL; no refill",
                              "unexpected_active_grad_failure": "STOP; declared dormant/initial-gamma/negative-only loc exceptions recorded"},
            "success_criteria": ["100 effective updates completed within <=1600 attempts and measured budget",
                                 "all finite losses/gradients and no hidden replacements/OOM/input identity drift",
                                 "checkpoint+same-partition resume validated on intended device and safe serial loader",
                                 "fixed-dev positive+safe-negative cohort available; final loss <=3x baseline; no accuracy claim"],
            "readiness": {"serial_loader": "pending", "windows_spawn2_diagnostic": "pending; not recommended for resume",
                          "cuda_synthetic_lifecycle": "pending GPU token", "real_forward_only_pipeline_budget": "pending lead phase",
                          "recommendation": "NO_GO until readiness evidence and budget bound; a GO would still NOT authorize real SGD"}}


def validate_frozen_pilot(config):
    template = pilot_recipe()
    # Policies are frozen before outcomes. Only these identity/evidence fields may be filled later.
    current, expected = json.loads(json.dumps(config)), json.loads(json.dumps(template))
    for value in (current, expected):
        for key in ("manifest", "manifest_sha256", "train_policy", "train_policy_sha256", "development_policy",
                    "development_policy_sha256", "reservations_sha256"):
            value["data"][key] = None
        value["development"]["cohort_manifest"] = value["development"]["cohort_sha256"] = None
        value["budget"]["max_wall_seconds"] = None
        value["budget"]["wall_budget_status"] = template["budget"]["wall_budget_status"]
        value["readiness"] = template["readiness"]
    if current != expected:
        raise ValueError("pilot predeclared policy/config drift")
    return canonical_hash(config)


def validate_temporal_binding(config):
    validate_frozen_pilot(config)
    data = config["data"]
    for path_key, hash_key in (("manifest", "manifest_sha256"), ("train_policy", "train_policy_sha256"),
                               ("development_policy", "development_policy_sha256")):
        if not data[path_key] or not data[hash_key] or sha256(ROOT/data[path_key]) != data[hash_key]:
            raise ValueError("temporal manifest/policy binding absent or changed")
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    dataset = TemporalCandidateDataset(ROOT/data["manifest"], ROOT/data["train_policy"], split="train",
                                       seed=config["seed"], length=config["budget"]["max_attempts"])
    if dataset.manifest_sha256 != data["manifest_sha256"] or dataset.policy_sha256 != data["train_policy_sha256"] or dataset.reservations_sha256 != data["reservations_sha256"]:
        raise ValueError("temporal indexedDataset identity mismatch")
    return dataset
