"""Owner-directed domain training: aerial vehicles + pedestrians, controlled AlexNet vs ResNet-18 comparison.

Same corrected B2 head, B2-L1 loss, data stream, schedule (source LR table, backbone unfreeze at epoch 10,
compressed epochs), gradient audit and clip for both --backbone choices; only the backbone differs.
Train tracks: LaSOT domain categories (training split), VisDrone-SOT train, VisDrone-MOT train (domain
classes), UAVDT (domain classes), mixed by --source-weights. DEV: fixed pairs from VisDrone-SOT/MOT val.
Primary checkpoint = final epoch. Tracking evaluation is separate (scripts/track_b2.py).

    python scripts/train_b2_domain.py --run-id DOMAIN-ALEX-001 --backbone alexnet --owner-instruction "<text>"
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
from run_b2_direct import CONFIG, Journal, digest, git  # noqa: E402

RESNET_WEIGHTS = ROOT/"artifacts/pretrained/resnet18-f37072fd.pth"
DEFAULT_WEIGHTS = {"LaSOT": 0.25, "VisDrone-SOT": 0.3, "VisDrone-MOT": 0.25, "UAVDT": 0.2}


ALL_TRAIN_PARTS = ("lasot", "visdrone-sot-1", "visdrone-sot-2", "visdrone-mot", "uavdt")


def domain_tracks(split, data_root=None, parts=ALL_TRAIN_PARTS, uavdt_holdout=()):
    """Train parts are selectable (a missing selected part is an error, never silently skipped).
    uavdt_holdout: UAVDT video names excluded from training (validation subset V3, scripts/validate_domain.py)."""
    from hift_research.lasot.tracks import lasot_track, uavdt_tracks, visdrone_mot_tracks, visdrone_sot_track
    data = Path(data_root or ROOT/"data")
    tracks, sot_roots, mot_roots = [], [], []
    if split == "train":
        unknown = set(parts)-set(ALL_TRAIN_PARTS)
        if unknown:
            raise ValueError(f"unknown train parts {unknown}")
        if "lasot" in parts:
            lasot = data/"lasot"
            train = set((lasot/"training_set.txt").read_text().split())
            tracks += [lasot_track(p) for p in sorted((lasot/"original").iterdir()) if p.name in train]
        sot_roots = [data/f"visdrone/SOT/VisDrone2019-SOT-train_part{i}/VisDrone2019-SOT-train" for i in (1, 2) if f"visdrone-sot-{i}" in parts]
        if "visdrone-mot" in parts:
            mot_roots = [data/"visdrone/MOT/VisDrone2019-MOT-train"]
        if "uavdt" in parts:
            uavdt = data/"uavdt"
            for seq in sorted((uavdt/"UAV-benchmark-M").iterdir()):
                gt = uavdt/"UAV-benchmark-MOTD_v1.0/GT"/f"{seq.name}_gt_whole.txt"
                if gt.is_file() and seq.name not in set(uavdt_holdout):
                    tracks += uavdt_tracks(seq, gt)
    else:
        sot_roots = [data/"visdrone/SOT/VisDrone2019-SOT-val/VisDrone2019-SOT-val"]
        mot_roots = [data/"visdrone/MOT/VisDrone2019-MOT-val"]
    for sot in sot_roots:
        for seq in sorted((sot/"sequences").iterdir()):
            tracks.append(visdrone_sot_track(seq, sot/"annotations"/f"{seq.name}.txt"))
    for mot_root in mot_roots:
        for seq in sorted((mot_root/"sequences").iterdir()):
            tracks += visdrone_mot_tracks(seq, mot_root/"annotations"/f"{seq.name}.txt")
    return tracks


def code_provenance():
    """Git revision + dirty-diff hash; on a cluster copy without .git, the shipped GIT_COMMIT file."""
    try:
        diff = git("diff", "--binary", "HEAD")
        return git("rev-parse", "HEAD").decode().strip(), hashlib.sha256(diff).hexdigest(), diff
    except Exception:
        marker = ROOT/"GIT_COMMIT"
        if not marker.is_file():
            raise RuntimeError("no git checkout and no GIT_COMMIT file: provenance unavailable")
        return marker.read_text().strip(), "clean-archive (git archive of a committed revision)", b""


def build_model(kind, resnet_weights=RESNET_WEIGHTS):
    if kind == "alexnet":
        from hift_research.training.graph import B2TrainingGraph
        from hift_research.training.lifecycle import EpochSchedule, apply_epoch_modes, lifecycle_optimizer
        from hift_research.training.recipe import load_legacy_backbone_strict
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        model = B2TrainingGraph()
        load_legacy_backbone_strict(model.backbone, ROOT/config["source_backbone_path"])
        modes = apply_epoch_modes
        optimizer = lifecycle_optimizer(model)
    else:
        from hift_research.backbones.resnet import B2ResNetGraph, apply_resnet_epoch_modes, load_imagenet_resnet18, resnet_optimizer
        from hift_research.training.lifecycle import EpochSchedule
        from hift_research.training.recipe import source_settings
        model = B2ResNetGraph()
        load_imagenet_resnet18(model.backbone, resnet_weights)
        modes = apply_resnet_epoch_modes
        optimizer = resnet_optimizer(model, source_settings()["TRAIN"]["BASE_LR"])
    modes(model, 0)
    return model, optimizer, EpochSchedule(optimizer), modes


class FinetuneSchedule:
    """Log-spaced grader LR from start to end over the run; backbone group at backbone_ratio of it."""
    def __init__(self, optimizer, epochs, start, end, backbone_ratio=0.1):
        import numpy as np
        self.optimizer, self.ratio = optimizer, backbone_ratio
        self.table = np.logspace(np.log10(start), np.log10(end), epochs).tolist()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.optimizer.param_groups[0]["lr"] = self.table[epoch]*self.ratio
        self.optimizer.param_groups[1]["lr"] = self.table[epoch]

    def state_dict(self):
        return {"epoch": self.epoch, "table": self.table, "backbone_ratio": self.ratio}


def build_official_finetune(layout, epochs, start, end, backbone="trainable"):
    """B0-FT: released HiFT checkpoint (hash-pinned, strict keys) as initialization.
    layout='original' keeps the released (scrambling) encoder computation the weights were learned with;
    'corrected' binds the B2 layout fix. backbone='trainable': layers3-5 (+their BN) train from step 0 (post-unfreeze
    source mode); 'frozen': whole backbone frozen with BN in eval mode (pre-unfreeze source mode)."""
    from hift_research import OFFICIAL_CHECKPOINT_SHA256
    from hift_research.inference import read_checkpoint_state
    from hift_research.provenance import sha256
    from hift_research.training.graph import B2TrainingGraph
    from hift_research.training.lifecycle import apply_epoch_modes, lifecycle_optimizer
    path = ROOT/"artifacts/checkpoints/general_model.pth"
    if sha256(path) != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("official checkpoint hash mismatch")
    model = B2TrainingGraph()
    model.load_state_dict(read_checkpoint_state(path), strict=True)
    if layout == "original":
        for layer in model.grader.transformer.encoder.layers:
            del layer.forward  # back to the upstream class method: the released computation
    if backbone not in ("trainable", "frozen"):
        raise ValueError("backbone must be trainable or frozen")
    source_epoch = 10 if backbone == "trainable" else 0
    def modes(m, epoch):
        return apply_epoch_modes(m, source_epoch)
    modes(model, 0)
    optimizer = lifecycle_optimizer(model)
    return model, optimizer, FinetuneSchedule(optimizer, epochs, start, end), modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backbone", choices=["alexnet", "resnet18"], required=True)
    parser.add_argument("--init", choices=["scratch-head", "official"], default="scratch-head",
                        help="official = fine-tune the released HiFT checkpoint (alexnet only)")
    parser.add_argument("--layout", choices=["corrected", "original"], default="corrected",
                        help="encoder layout; 'original' only meaningful with --init official")
    parser.add_argument("--ft-lr", nargs=2, type=float, default=[1e-3, 1e-4], help="fine-tune grader LR start end")
    parser.add_argument("--ft-backbone", choices=["trainable", "frozen"], default="trainable",
                        help="fine-tune: backbone layers3-5 trainable from step 0 (default, as before) or frozen")
    parser.add_argument("--distractor-probability", type=float, default=0.0,
                        help="stage-2 P2: share of pairs that are same-video same-group distractor negatives (default 0 = none)")
    parser.add_argument("--max-gap", type=int, default=100, help="positive pairs: max template-search frame gap (default 100)")
    parser.add_argument("--long-gap-probability", type=float, default=0.0,
                        help="stage-4 S4-GAP: share of positive pairs whose search frame is drawn from the whole track (default 0)")
    parser.add_argument("--uavdt-holdout-fraction", type=float, default=0.0,
                        help="hold whole UAVDT videos out of training for validation V3 (default 0 = none, as before)")
    parser.add_argument("--owner-instruction", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=20261027)
    parser.add_argument("--seed", type=int, default=None, help="model/device seed (default: the frozen config seed); seed replicates")
    parser.add_argument("--source-weights", type=json.loads, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-root", type=Path, default=ROOT/"data")
    parser.add_argument("--train-parts", nargs="+", default=list(ALL_TRAIN_PARTS), choices=ALL_TRAIN_PARTS)
    parser.add_argument("--output-root", type=Path, default=ROOT/"artifacts"/"b2-domain")
    parser.add_argument("--resnet-weights", type=Path, default=RESNET_WEIGHTS)
    parser.add_argument("--search-scale-aug", type=float, default=0.18,
                        help="search-crop scale augmentation (source pysot 0.18); step 3 tests larger values")
    args = parser.parse_args()
    output = args.output_root/args.run_id
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from torch.utils.data import DataLoader
    from hift_research.provenance import sha256
    from hift_research.lasot.dataset import TrackPairDataset, collate_pairs
    from hift_research.corrections.regression_l1 import window_loss_with_l1
    from hift_research.training.graph import gradient_audit
    from hift_research.training.lifecycle import actual_heads, window_denominators, window_loss
    from hift_research.training.wave4_lifecycle import cpu_tree, device_tree, seed_device

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    from hift_research.evaluation.validation import uavdt_holdout
    holdout = []
    if args.uavdt_holdout_fraction and "uavdt" in args.train_parts:
        gt_dir = args.data_root/"uavdt/UAV-benchmark-MOTD_v1.0/GT"
        holdout = uavdt_holdout([p.name for p in (args.data_root/"uavdt/UAV-benchmark-M").iterdir()
                                 if (gt_dir/f"{p.name}_gt_whole.txt").is_file()], args.uavdt_holdout_fraction)
    train_tracks = domain_tracks("train", args.data_root, args.train_parts, uavdt_holdout=holdout)
    val_tracks = domain_tracks("val", args.data_root)
    weights = {k: v for k, v in args.source_weights.items() if any(t["source"] == k for t in train_tracks)}
    total = args.epochs*args.steps_per_epoch*args.batch
    dataset = TrackPairDataset(train_tracks, length=total, seed=args.data_seed, source_weights=weights,
                               search_scale=args.search_scale_aug, distractor_probability=args.distractor_probability,
                               max_gap=args.max_gap, long_gap_probability=args.long_gap_probability)
    dev_set = TrackPairDataset(val_tracks, length=512, seed=20261028, negative_probability=0.5)
    dev_batches = [collate_pairs([dev_set[i] for i in range(s, s+32)]) for s in range(0, 512, 32)]
    dev_batches = [b for b in dev_batches if b.get("template") is not None]
    seed = config["seed"] if args.seed is None else args.seed
    device = seed_device(seed, "cuda:0", gpu_token="owner-domain-"+args.run_id)
    torch.use_deterministic_algorithms(False)
    if args.init == "official":
        if args.backbone != "alexnet":
            raise ValueError("the released checkpoint is AlexNet-HiFT")
        model, optimizer, scheduler, modes = build_official_finetune(args.layout, args.epochs, *args.ft_lr, backbone=args.ft_backbone)
    else:
        if args.layout != "corrected":
            raise ValueError("scratch-head training always uses the corrected layout")
        model, optimizer, scheduler, modes = build_model(args.backbone, args.resnet_weights)
    init = digest(model.named_parameters())
    model = model.to(device)
    train_loss = functools.partial(window_loss_with_l1, l1_weight=args.l1_weight) if args.l1_weight else window_loss

    def dev_loss():
        model.eval()
        gpu = [device_tree({k: b[k] for k in ("template", "search", "targets")}, device) for b in dev_batches]
        denominators = window_denominators(gpu)
        with torch.inference_mode():
            value = sum(float(window_loss(actual_heads(model, g), g["targets"], denominators)) for g in gpu)
        return value

    journal = Journal(output/"journal.jsonl")
    revision, diff_hash, diff = code_provenance()
    (output/"dirty.diff").write_bytes(diff)
    counts = {}
    for t in train_tracks:
        key = f"{t['source']}/{t['category']}"
        counts[key] = counts.get(key, 0)+1
    ledgers = [p for p in (ROOT/"artifacts/lasot/ACQUISITION-001.json", ROOT/"artifacts/visdrone/MOT-ACQUISITION-001.json",
                           ROOT/"artifacts/visdrone/SOT-ACQUISITION-001.json", ROOT/"artifacts/uavdt/ACQUISITION-001.json") if p.exists()]
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "hift-b2-domain-run-v1", "run_id": args.run_id, "backbone": args.backbone,
                   "init": args.init, "layout": args.layout,
                   "identity": ("B0-FT (released checkpoint fine-tuned; " + args.layout + " layout)") if args.init == "official" else "B2-domain",
                   "variant": f"B2-L1(weight={args.l1_weight}) head + {args.backbone} backbone; domain = aerial vehicles + pedestrians",
                   "owner_instruction": args.owner_instruction,
                   "data": {"data_root": str(args.data_root), "train_parts": args.train_parts,
                            "train_tracks_by_source_category": counts, "train_tracks": len(dataset.sequences),
                            "val_tracks": len(dev_set.sequences), "source_weights": weights, "pairs_total": total,
                            "search_augmentation": dataset.search_aug,
                            "data_seed": args.data_seed, "uavdt_holdout_fraction": args.uavdt_holdout_fraction,
                            "uavdt_holdout_videos": holdout, "distractor_probability": args.distractor_probability, "max_gap": args.max_gap, "long_gap_probability": args.long_gap_probability, "acquisition_ledgers": {str(p): sha256(p) for p in ledgers},
                            "test_sets_excluded": "UAV123 (all), VisDrone-SOT test-dev, VisDrone-MOT test-dev never used in training"},
                   "schedule": {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch": args.batch,
                                "lr_table_used": scheduler.table[:args.epochs],
                                "backbone_mode": (f"fine-tune, backbone {args.ft_backbone}" + (" (layers3-5 + their BN train from step 0)"
                                                  if args.ft_backbone == "trainable" else " (all frozen, BN eval)"))
                                                 if args.init == "official" else "source schedule: layers unfreeze at source epoch 10",
                                "deterministic_algorithms": False},
                   "backbone_weights": ("released HiFT checkpoint (hash-pinned)" if args.init == "official" else
                                        "third_party/HiFT/pretrained_models/alexnet-bn.pth" if args.backbone == "alexnet" else
                                        f"{args.resnet_weights.name} sha256 {sha256(args.resnet_weights)}"),
                   "initial_parameters_sha256": init, "git_revision": revision,
                   "dirty_diff_sha256": diff_hash, "seed": seed, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    journal.emit("start", train_tracks=len(dataset.sequences), val_tracks=len(dev_set.sequences), dev_batches=len(dev_batches))

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_pairs,
                        persistent_workers=True, prefetch_factor=4)
    results, status, origin, step, rejected = [], "completed", time.perf_counter(), 0, 0
    batches = iter(loader)
    try:
        results.append({"epoch": -1, "step": 0, "dev_total_loss": dev_loss()})
        journal.emit("dev", **results[-1])
        for epoch in range(args.epochs):
            scheduler.set_epoch(epoch); modes(model, epoch)
            window, window_start = [], time.perf_counter()
            for _ in range(args.steps_per_epoch):
                batch = next(batches)
                rejected += sum(1 for r in batch["records"] if "reason" in r)
                if batch.get("template") is None:
                    continue
                gpu = device_tree({k: batch[k] for k in ("template", "search", "targets")}, device)
                optimizer.zero_grad(set_to_none=True)
                loss = train_loss(actual_heads(model, gpu), gpu["targets"], window_denominators([gpu]))
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
                                 steps_per_s=len(window)/(time.perf_counter()-window_start))
                    window, window_start = [], time.perf_counter()
            value = dev_loss()
            modes(model, epoch)
            path = output/f"epoch-{epoch:02d}.pt"
            with path.open("xb") as stream:
                torch.save({"schema": "hift-b2-domain-checkpoint-v1", "backbone": args.backbone, "layout": args.layout,
                            "init": args.init, "update": step, "epoch": epoch,
                            "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                            "scheduler": scheduler.state_dict()}, stream)
            results.append({"epoch": epoch, "step": step, "dev_total_loss": value})
            journal.emit("epoch_end", epoch=epoch, step=step, dev_total_loss=value,
                         checkpoint={"path": str(path), "sha256": sha256(path)}, elapsed_s=time.perf_counter()-origin)
    except Exception as exc:
        status = f"failed:{type(exc).__name__}:{exc}"
        journal.emit("failure", error=status, step=step)
    if status == "completed":  # model-only copy of the fixed-last checkpoint, small enough to retrieve over a VPN
        final = output/"final-model.pt"
        with final.open("xb") as stream:
            torch.save({"schema": "hift-b2-domain-checkpoint-v1", "backbone": args.backbone, "layout": args.layout,
                        "init": args.init, "update": step, "epoch": args.epochs-1, "model": cpu_tree(model.state_dict())}, stream)
        journal.emit("final_model", path=str(final), sha256=sha256(final))
    summary = {"status": status, "steps": step, "rejected_pairs": rejected, "wall_seconds": time.perf_counter()-origin,
               "development": results, "primary_checkpoint": f"epoch-{args.epochs-1:02d}.pt (fixed last)",
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    journal.emit("finished", status=status, steps=step, wall_seconds=summary["wall_seconds"])
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
