"""Owner-directed direct B2 pilot: frozen Wave4 recipe without the Wave6/7 guarded launcher.

Same config, data, seed, initialization, SerialPilotController (byte-unchanged), loss,
gradient audit, clipping, SGD and fixed DEV0/25/50/100 cohort. Omitted: the 900 s ledger,
parent resource monitor, Windows Job containment and per-operation source rehashing.
Added: journal of the full gradient audit and per-parameter norms if the audit fails.
Optional --loc-l1-weight selects the separate B2-L1 recipe variant (see corrections/regression_l1.py).

    python scripts/run_b2_direct.py --run-id B2-DIRECT-001 --owner-instruction "<text>" [--loc-l1-weight 1.0]
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
CONFIG = ROOT/"configs"/"B2-Pilot-W4.json"
INIT_REFERENCE = ROOT/"artifacts/memory-contract-continuation/OWNER-BOUND-PROPOSAL.json"
UPSTREAM_SHA = "992aa7c1605360d607af84cb804b93cd8ee81caf"


def digest(items):
    h = hashlib.sha256()
    for name, tensor in items:
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


class Journal:
    def __init__(self, path):
        self.stream = path.open("x", encoding="utf-8", newline="\n")

    def emit(self, event, **fields):
        row = {"event": event, "time": time.time(), **fields}
        self.stream.write(json.dumps(row, allow_nan=False, default=str)+"\n"); self.stream.flush(); os.fsync(self.stream.fileno())
        print(json.dumps(row, default=str)[:300], flush=True)


def development(model, dev, window_loss, actual_heads, device_tree, apply_epoch_modes, buffered, device):
    import torch
    batches, denominators = buffered(dev, list(range(128)), kind="real_development_forward_only")
    records = [r for b in batches for r in b["accepted_records"]]
    positive = sum(not r["candidate_negative"] for r in records)
    if (positive, len(records)-positive) != (64, 64) or any(b["rejected_records"] for b in batches):
        raise ValueError("fixed 64 positive / 64 negative DEV cohort incomplete")
    total = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                batch = device_tree(batch, device)
                value = window_loss(actual_heads(model, batch), batch["targets"], denominators)
                if not bool(torch.isfinite(value)):
                    raise ValueError("nonfinite DEV loss")
                total += float(value)
    finally:
        apply_epoch_modes(model, 0)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-instruction", required=True)
    parser.add_argument("--loc-l1-weight", type=float, default=0.0,
                        help="0 = B2 exactly; >0 = B2-L1 variant (raw-mapping L1 localization term)")
    args = parser.parse_args()
    if not args.loc_l1_weight >= 0:
        raise ValueError("--loc-l1-weight must be nonnegative")
    variant = "B2" if args.loc_l1_weight == 0 else f"B2-L1(weight={args.loc_l1_weight})"
    output = ROOT/"artifacts"/"b2-direct"/args.run_id
    output.mkdir(parents=True, exist_ok=False)  # never overwrite a run

    import torch
    from hift_research.provenance import sha256
    from hift_research.training import wave4_pilot as legacy
    from hift_research.training import wave4_pilot_engine as engine
    from hift_research.training.graph import B2TrainingGraph, gradient_audit
    from hift_research.training.lifecycle import actual_heads, apply_epoch_modes, window_loss
    from hift_research.training.recipe import load_legacy_backbone_strict
    from hift_research.training.wave4_lifecycle import device_tree, seed_device
    from hift_research.training.wave4_loader import dataset_identity
    from hift_research.training.wave4_recipes import validate_temporal_binding
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    from hift_research.corrections.regression_l1 import window_loss_with_l1
    import functools
    train_loss = window_loss if variant == "B2" else functools.partial(window_loss_with_l1, l1_weight=args.loc_l1_weight)

    journal = Journal(output/"journal.jsonl")
    diff = git("diff", "--binary", "HEAD")
    (output/"dirty.diff").write_bytes(diff)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset = validate_temporal_binding(config)
    dev = TemporalCandidateDataset(ROOT/config["data"]["manifest"], ROOT/config["data"]["development_policy"],
                                   split="development", seed=20261024, length=128)
    token = "owner-direct-"+args.run_id
    device = seed_device(config["seed"], "cuda:0", gpu_token=token) or torch.device("cuda:0")
    model = B2TrainingGraph()
    load_legacy_backbone_strict(model.backbone, ROOT/config["source_backbone_path"])
    apply_epoch_modes(model, 0)
    init = {"all_parameters_sha256": digest(model.named_parameters()),
            "backbone_state_sha256": digest(model.backbone.state_dict().items()),
            "grader_parameters_sha256": digest(model.grader.named_parameters())}
    expected = json.loads(INIT_REFERENCE.read_text(encoding="utf-8"))["scope"]["initialization"]
    if init != expected:
        raise ValueError("seeded initialization differs from the frozen pilot initialization")
    identity = dataset_identity(dataset); identity["kind"] = "authorized_future_real"
    manifest = {
        "schema": "hift-b2-direct-run-v1", "run_id": args.run_id, "variant": variant,
        "variant_note": "B2 = corrected training pilot. B2-L1 adds a raw-mapping L1 localization term (recipe change, separate identity).",
        "owner_instruction": args.owner_instruction, "launcher": "scripts/run_b2_direct.py (not the Wave6/7 guarded launcher)",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git("rev-parse", "HEAD").decode().strip(), "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "upstream_sha": git("-C", "third_party/HiFT", "rev-parse", "HEAD").decode().strip(),
        "upstream_clean": not git("-C", "third_party/HiFT", "status", "--porcelain"),
        "config": {"path": str(CONFIG), "sha256": sha256(CONFIG)}, "seed": config["seed"], "dev_seed": 20261024,
        "initialization": init, "data_identity": identity, "development_identity": dataset_identity(dev),
        "recipe": {"physical_batch": 4, "attempts_per_window": 16, "max_updates": 100, "max_attempts": 1600,
                   "clip_norm": 10.0, "dev_updates": [0, 25, 50, 100], "learning_signal_criterion": "DEV100/DEV0 <= 0.95"},
        "software": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "uuid": str(torch.cuda.get_device_properties(0).uuid)},
        "omitted_safeguards": ["900 s inclusive ledger", "parent NVML/CUDA resource monitor", "Windows Job containment",
                               "per-operation source/data rehash", "GPU lease"],
        "limitations": ["Head-only pilot on frozen ImageNet backbone; not checkpoint reproduction or tracking accuracy",
                        "Other Windows clients may share the GPU"]}
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, default=str))
    journal.emit("start", run_id=args.run_id, initialization_matches=True)

    failures = {}
    def recording_audit(m):
        audit = gradient_audit(m)
        if audit["unexpected_zero"] or audit["unexpected_missing"] or audit["nonfinite"]:
            flagged = set(audit["unexpected_zero"]) | set(audit["unexpected_missing"]) | set(audit["nonfinite"])
            failures.update(audit)
            journal.emit("gradient_audit_failure", **{k: audit[k] for k in ("unexpected_zero", "unexpected_missing", "nonfinite")},
                         flagged_norms={n: (None if p.grad is None else float(p.grad.norm())) for n, p in m.named_parameters() if n in flagged},
                         gammas={n: float(p.detach()) for n, p in m.named_parameters() if n.endswith("gamma")})
        return audit

    def bridge(path, authorization_file=None, *, ack_real_sgd=False):
        return config, {"exclusive_gpu_token": token}

    results = []
    def evaluate(controller):
        # dev_total_loss is always the prespecified B2 objective; the variant objective is reported separately.
        loss, variant_loss = (development(controller.model, dev, objective, actual_heads, device_tree, apply_epoch_modes,
                                          legacy.buffered_attempt_window, controller.device) for objective in (window_loss, train_loss))
        results.append({"update": controller.update, "dev_total_loss": loss, "dev_training_objective": variant_loss})
        receipt = controller.save_boundary(output/f"update-{controller.update:04d}.pt")
        journal.emit("development", update=controller.update, dev_total_loss=loss, dev_training_objective=variant_loss, checkpoint=receipt)

    origin = time.perf_counter()
    status = "completed"
    with patch.object(legacy, "future_authorization", bridge), patch.object(engine, "gradient_audit", recording_audit), \
            patch.object(engine, "window_loss", train_loss):
        controller = engine.SerialPilotController(model, dataset, config_identity={"config_sha256": sha256(CONFIG)},
            data_identity=identity, device=device, gpu_token=token, synthetic=False, physical=4, attempts_per_window=16,
            max_updates=100, max_attempts=1600, config_path=CONFIG, authorization_file=output/"manifest.json", ack_real_sgd=True)
        evaluate(controller)
        try:
            while controller.cursor < 1600 and controller.update < 100 and controller.window < 100:
                result = controller.step()
                if result is None:
                    break
                s, m = result["semantic"], result["measurement"]
                journal.emit("update", update=s["update"], window=s["window"], loss=s["loss"], norm_before_clip=s["norm_before_clip"],
                             accepted=s["accepted"], rejected=s["rejected"], window_s=m["timings"]["window_total_s"],
                             peak_reserved=(m["memory"] or {}).get("peak_reserved"))
                if s["stepped"] and controller.update in (25, 50, 100):
                    evaluate(controller)
        except Exception as exc:
            status = f"failed:{type(exc).__name__}:{exc}"
            journal.emit("failure", error=status, update=controller.update, cursor=controller.cursor)
    dev0 = results[0]["dev_total_loss"]
    final = next((r for r in results if r["update"] == 100), None)
    summary = {"status": status, "updates": controller.update, "attempts": controller.cursor, "wall_seconds": time.perf_counter()-origin,
               "development": results, "dev100_over_dev0": None if final is None else final["dev_total_loss"]/dev0,
               "learning_signal": None if final is None else final["dev_total_loss"]/dev0 <= 0.95,
               "gradient_audit_failure": failures or None, "finished_utc": datetime.now(timezone.utc).isoformat()}
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, default=str))
    journal.emit("finished", **{k: summary[k] for k in ("status", "updates", "wall_seconds", "dev100_over_dev0", "learning_signal")})
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
