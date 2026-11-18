"""Owner-directed B2-L1 training on official LaSOT training-split sequences (fresh seeded init).

Recipe: B2 corrected graph + B2-L1 loss, source optimizer groups, source LR table and backbone
unfreeze (layers3-5 at source epoch 10) for epochs 0..E-1, each epoch compressed to
--steps-per-epoch steps of --batch pairs (NOT the 600k-attempt source epoch). Every index is a
unique pair (no repetition). Gradient audit + finite checks + clip 10 each step. MOT17 fixed
DEV loss after each epoch (cross-domain diagnostic). Checkpoint per epoch; primary = final.

    python scripts/train_b2_lasot.py --run-id B2-L1-LASOT-001 --epochs 20 --owner-instruction "<text>"
"""
import argparse
import functools
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from run_b2_direct import CONFIG, INIT_REFERENCE, Journal, development, digest, git  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-instruction", required=True)
    parser.add_argument("--lasot-root", type=Path, default=ROOT/"data/lasot")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=20261026)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 70 or args.steps_per_epoch < 1 or args.batch < 2 or not args.l1_weight >= 0:
        raise ValueError("invalid epochs/steps/batch/l1-weight")
    output = ROOT/"artifacts"/"b2-lasot"/args.run_id
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from torch.utils.data import DataLoader
    from hift_research.provenance import sha256
    from hift_research.lasot.dataset import LaSOTPairDataset, collate_pairs
    from hift_research.corrections.regression_l1 import window_loss_with_l1
    from hift_research.training import wave4_pilot as legacy
    from hift_research.training.graph import B2TrainingGraph, gradient_audit
    from hift_research.training.lifecycle import (EpochSchedule, actual_heads, apply_epoch_modes, lifecycle_optimizer,
                                                  window_denominators, window_loss)
    from hift_research.training.recipe import load_legacy_backbone_strict
    from hift_research.training.wave4_lifecycle import cpu_tree, device_tree, seed_device
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    training = set((args.lasot_root/"training_set.txt").read_text().split())
    testing = set((args.lasot_root/"testing_set.txt").read_text().split())
    extracted = {p.name for p in (args.lasot_root/"original").iterdir() if p.is_dir()}
    if extracted & testing:
        raise ValueError("LaSOT testing_set sequences present in training root")
    sequences = sorted(extracted & training)
    total = args.epochs*args.steps_per_epoch*args.batch
    dataset = LaSOTPairDataset(args.lasot_root, sequences, length=total, seed=args.data_seed)
    dev = TemporalCandidateDataset(ROOT/config["data"]["manifest"], ROOT/config["data"]["development_policy"],
                                   split="development", seed=20261024, length=128)
    device = seed_device(config["seed"], "cuda:0", gpu_token="owner-lasot-"+args.run_id)
    torch.use_deterministic_algorithms(False)  # throughput run; seeds recorded, bitwise replay not claimed
    model = B2TrainingGraph()
    load_legacy_backbone_strict(model.backbone, ROOT/config["source_backbone_path"])
    apply_epoch_modes(model, 0)
    init = {"all_parameters_sha256": digest(model.named_parameters()),
            "backbone_state_sha256": digest(model.backbone.state_dict().items()),
            "grader_parameters_sha256": digest(model.grader.named_parameters())}
    if init != json.loads(INIT_REFERENCE.read_text(encoding="utf-8"))["scope"]["initialization"]:
        raise ValueError("seeded initialization differs from the frozen pilot initialization")
    model = model.to(device)
    optimizer = lifecycle_optimizer(model)
    scheduler = EpochSchedule(optimizer)
    train_loss = functools.partial(window_loss_with_l1, l1_weight=args.l1_weight) if args.l1_weight else window_loss

    journal = Journal(output/"journal.jsonl")
    diff = git("diff", "--binary", "HEAD")
    (output/"dirty.diff").write_bytes(diff)
    acquisition = ROOT/"artifacts/lasot/ACQUISITION-001.json"
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "hift-b2-lasot-run-v1", "run_id": args.run_id, "variant": f"B2-L1(weight={args.l1_weight})" if args.l1_weight else "B2",
                   "owner_instruction": args.owner_instruction, "data": {"dataset": "LaSOT (official l-lt/LaSOT, training_set.txt only)",
                   "sequences": sequences, "sequence_count": len(dataset.sequences), "pairs_total": total, "data_seed": args.data_seed,
                   "acquisition_ledger": {"path": str(acquisition), "sha256": sha256(acquisition)} if acquisition.exists() else "pending",
                   "negative_probability": dataset.negative_probability, "max_gap": dataset.max_gap},
                   "schedule": {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch": args.batch,
                                "lr_table_used": scheduler.table[:args.epochs], "backbone_unfreeze_epoch": 10,
                                "note": "source epoch indices with compressed epochs; not the 70x600k source schedule"},
                   "initialization": init, "git_revision": git("rev-parse", "HEAD").decode().strip(),
                   "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(), "upstream_sha": git("-C", "third_party/HiFT", "rev-parse", "HEAD").decode().strip(),
                   "config_sha256": sha256(CONFIG), "seed": config["seed"], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "determinism": "nondeterministic algorithms allowed; worker order fixed by index", "started_utc": datetime.now(timezone.utc).isoformat()},
                  stream, indent=2)
    journal.emit("start", sequences=len(dataset.sequences), pairs_total=total)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_pairs,
                        persistent_workers=True, prefetch_factor=4)
    results, status, origin, step, rejected = [], "completed", time.perf_counter(), 0, 0
    batches = iter(loader)
    try:
        for epoch in range(args.epochs):
            scheduler.set_epoch(epoch); apply_epoch_modes(model, epoch)
            window, window_start = [], time.perf_counter()
            for _ in range(args.steps_per_epoch):
                batch = next(batches)
                rejected += sum(1 for r in batch["records"] if "reason" in r)
                if batch.get("template") is None:
                    continue
                gpu = device_tree({k: batch[k] for k in ("template", "search", "targets")}, device)
                optimizer.zero_grad(set_to_none=True)
                loss = train_loss(actual_heads(model, gpu), gpu["targets"], window_denominators([gpu]))  # same tensors the loss sees
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite loss")
                loss.backward()
                audit = gradient_audit(model)
                if audit["unexpected_missing"] or audit["nonfinite"] or audit["unexpected_zero"]:
                    journal.emit("gradient_audit_failure", step=step, **{k: audit[k] for k in ("unexpected_zero", "unexpected_missing", "nonfinite")})
                    raise ValueError("unexpected active-gradient failure")
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
                step += 1
                window.append((float(loss.detach()), norm))
                if step % 100 == 0:
                    journal.emit("progress", epoch=epoch, step=step, mean_loss=sum(l for l, _ in window)/len(window),
                                 mean_norm=sum(n for _, n in window)/len(window), rejected_total=rejected,
                                 steps_per_s=len(window)/(time.perf_counter()-window_start), lr=[g["lr"] for g in optimizer.param_groups])
                    window, window_start = [], time.perf_counter()
            loss_dev, objective = (development(model, dev, f, actual_heads, device_tree, apply_epoch_modes,
                                               legacy.buffered_attempt_window, device) for f in (window_loss, train_loss))
            path = output/f"epoch-{epoch:02d}.pt"
            state = {"schema": "hift-b2-lasot-checkpoint-v1", "update": step, "epoch": epoch, "model": cpu_tree(model.state_dict()),
                     "optimizer": cpu_tree(optimizer.state_dict()), "scheduler": scheduler.state_dict()}
            with path.open("xb") as stream:
                torch.save(state, stream)
            results.append({"epoch": epoch, "step": step, "dev_total_loss": loss_dev, "dev_training_objective": objective})
            journal.emit("epoch_end", epoch=epoch, step=step, dev_total_loss=loss_dev, dev_training_objective=objective,
                         checkpoint={"path": str(path), "sha256": sha256(path)}, elapsed_s=time.perf_counter()-origin)
    except Exception as exc:
        status = f"failed:{type(exc).__name__}:{exc}"
        journal.emit("failure", error=status, step=step)
    summary = {"status": status, "steps": step, "rejected_pairs": rejected, "wall_seconds": time.perf_counter()-origin,
               "development": results, "primary_checkpoint": f"epoch-{args.epochs-1:02d}.pt (fixed last)",
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    journal.emit("finished", status=status, steps=step, wall_seconds=summary["wall_seconds"])
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
