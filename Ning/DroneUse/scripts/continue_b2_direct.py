"""Owner-directed multi-epoch continuation of a run_b2_direct.py run from a saved boundary.

Restores model/optimizer/momentum/RNG via the byte-unchanged controller's strict restore, then
repeats the SAME frozen 1600-attempt training cohort for further epochs (identical indexed
pairs each pass; the frozen policy allows no more data). Same head-only recipe, epoch-0 LR,
loss variant and gradient audit as the source run. DEV loss each epoch; checkpoints every
--checkpoint-every epochs and at the end. Not a source-schedule (70-epoch) reproduction.

    python scripts/continue_b2_direct.py --source-run B2-L1-001 --run-id B2-L1-001-E30 --epochs 30 --owner-instruction "<text>"
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from run_b2_direct import CONFIG, INIT_REFERENCE, Journal, development, digest, git  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Total passes over the 1600-attempt cohort, including the source run's first")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--owner-instruction", required=True)
    args = parser.parse_args()
    if not 2 <= args.epochs <= 200 or args.checkpoint_every < 1:
        raise ValueError("epochs must be in [2,200] and checkpoint-every >= 1")
    source = ROOT/"artifacts"/"b2-direct"/args.source_run
    source_manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    boundary = source/"update-0100.pt"
    output = ROOT/"artifacts"/"b2-direct"/args.run_id
    output.mkdir(parents=True, exist_ok=False)

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

    variant = source_manifest["variant"]
    weight = 0.0 if variant == "B2" else float(variant.split("weight=")[1].rstrip(")"))
    train_loss = window_loss if weight == 0 else functools.partial(window_loss_with_l1, l1_weight=weight)
    journal = Journal(output/"journal.jsonl")
    diff = git("diff", "--binary", "HEAD")
    (output/"dirty.diff").write_bytes(diff)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset = validate_temporal_binding(config)
    dev = TemporalCandidateDataset(ROOT/config["data"]["manifest"], ROOT/config["data"]["development_policy"],
                                   split="development", seed=20261024, length=128)
    token = "owner-direct-"+args.run_id
    device = seed_device(config["seed"], "cuda:0", gpu_token=token)
    model = B2TrainingGraph()
    load_legacy_backbone_strict(model.backbone, ROOT/config["source_backbone_path"])
    apply_epoch_modes(model, 0)
    identity = dataset_identity(dataset); identity["kind"] = "authorized_future_real"
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "hift-b2-direct-continuation-v1", "run_id": args.run_id, "variant": variant,
                   "source_run": args.source_run, "source_boundary": {"path": str(boundary), "sha256": sha256(boundary)},
                   "owner_instruction": args.owner_instruction, "epochs_total": args.epochs,
                   "data": "same frozen 1600-attempt MOT17 cohort repeated each epoch (identical indexed pairs)",
                   "schedule": "epoch-0 LR held constant, head-only (backbone frozen); not the 70-epoch source schedule",
                   "git_revision": git("rev-parse", "HEAD").decode().strip(), "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                   "config_sha256": sha256(CONFIG), "seed": config["seed"], "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)

    failures = {}
    def recording_audit(m):
        audit = gradient_audit(m)
        if audit["unexpected_zero"] or audit["unexpected_missing"] or audit["nonfinite"]:
            failures.update(audit)
            journal.emit("gradient_audit_failure", **{k: audit[k] for k in ("unexpected_zero", "unexpected_missing", "nonfinite")})
        return audit

    def bridge(path, authorization_file=None, *, ack_real_sgd=False):
        return config, {"exclusive_gpu_token": token}

    results, status, origin = [], "completed", time.perf_counter()
    with patch.object(legacy, "future_authorization", bridge), patch.object(engine, "gradient_audit", recording_audit), \
            patch.object(engine, "window_loss", train_loss):
        controller = engine.SerialPilotController(model, dataset, config_identity={"config_sha256": sha256(CONFIG)},
            data_identity=identity, device=device, gpu_token=token, synthetic=False, physical=4, attempts_per_window=16,
            max_updates=100, max_attempts=1600, config_path=CONFIG, authorization_file=source/"manifest.json", ack_real_sgd=True)
        controller.restore_boundary(boundary, expected_sha256=sha256(boundary))
        controller.max_updates = 100*args.epochs  # repeat the cohort; the controller's per-pass cursor limit stays 1600
        def evaluate(epoch, save):
            loss, objective = (development(controller.model, dev, f, actual_heads, device_tree, apply_epoch_modes,
                                           legacy.buffered_attempt_window, controller.device) for f in (window_loss, train_loss))
            results.append({"epoch": epoch, "update": controller.update, "dev_total_loss": loss, "dev_training_objective": objective})
            receipt = controller.save_boundary(output/f"update-{controller.update:05d}.pt") if save else None
            journal.emit("development", epoch=epoch, update=controller.update, dev_total_loss=loss,
                         dev_training_objective=objective, checkpoint=receipt)
        evaluate(1, save=False)
        try:
            for epoch in range(2, args.epochs+1):
                controller.cursor = 0
                losses = []
                while (result := controller.step()) is not None:
                    losses.append(result["semantic"]["loss"])
                journal.emit("epoch", epoch=epoch, update=controller.update, mean_train_loss=sum(losses)/len(losses),
                             windows=len(losses), seconds=time.perf_counter()-origin)
                evaluate(epoch, save=epoch % args.checkpoint_every == 0 or epoch == args.epochs)
        except Exception as exc:
            status = f"failed:{type(exc).__name__}:{exc}"
            journal.emit("failure", error=status, update=controller.update, cursor=controller.cursor)
    dev0 = json.loads((source/"summary.json").read_text(encoding="utf-8"))["development"][0]["dev_total_loss"]
    summary = {"status": status, "updates": controller.update, "wall_seconds": time.perf_counter()-origin,
               "dev0_from_source": dev0, "development": results,
               "final_dev_over_dev0": results[-1]["dev_total_loss"]/dev0,
               "best_dev_epoch": min(results, key=lambda r: (r["dev_total_loss"], r["epoch"]))["epoch"],
               "gradient_audit_failure": failures or None, "finished_utc": datetime.now(timezone.utc).isoformat()}
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    journal.emit("finished", **{k: summary[k] for k in ("status", "updates", "wall_seconds", "final_dev_over_dev0", "best_dev_epoch")})
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
