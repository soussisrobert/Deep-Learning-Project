"""Validation tracking (V1 VisDrone-SOT-val, V2 VisDrone-MOT-val episodes, V3 held-out UAVDT episodes) for
model selection. The sealed test sets (UAV123 domain, VisDrone-SOT test-dev) are never read here.
Ground truth initializes each episode and scores it afterwards; the tracker never sees later GT.

    python scripts/validate_domain.py --checkpoint runs/X/final-model.pt --data-root ~/hift/data --output runs/X/validation
    python scripts/validate_domain.py --official --score raw --selection upstream --data-root ... --output ...
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))

V2_PER_GROUP = {"vehicle": 40, "person": 40}
V3_PER_GROUP = {"vehicle": 40}
MAX_FRAMES = 200
MAX_PER_VIDEO = 7


def prefetched(pool, read, paths, depth=12):
    """Decode frames in background threads, at most ``depth`` ahead (bounded memory for long sequences)."""
    from collections import deque
    queue, it = deque(), iter(paths)
    for path in it:
        queue.append(pool.submit(read, path))
        if len(queue) >= depth:
            break
    while queue:
        image = queue.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            queue.append(pool.submit(read, nxt))
        yield image


def build_episodes(data_root, subsets, holdout_fraction):
    from hift_research.evaluation.validation import select_episodes, uavdt_holdout, make_episode
    from hift_research.lasot.tracks import uavdt_tracks, visdrone_mot_tracks, visdrone_sot_track
    data = Path(data_root)
    episodes, holdout = [], []
    if "V1" in subsets:
        sot = data/"visdrone/SOT/VisDrone2019-SOT-val/VisDrone2019-SOT-val"
        for seq in sorted((sot/"sequences").iterdir()):
            episode = make_episode(visdrone_sot_track(seq, sot/"annotations"/f"{seq.name}.txt"), max_frames=10**6)
            if episode is not None:
                episodes.append(("V1", episode))
    if "V2" in subsets:
        mot = data/"visdrone/MOT/VisDrone2019-MOT-val"
        tracks = [t for seq in sorted((mot/"sequences").iterdir())
                  for t in visdrone_mot_tracks(seq, mot/"annotations"/f"{seq.name}.txt")]
        episodes += [("V2", e) for e in select_episodes(tracks, per_group=V2_PER_GROUP, max_frames=MAX_FRAMES,
                                                         max_per_video=MAX_PER_VIDEO)]
    if "V3" in subsets:
        uavdt = data/"uavdt"
        videos = sorted(p.name for p in (uavdt/"UAV-benchmark-M").iterdir()
                        if (uavdt/"UAV-benchmark-MOTD_v1.0/GT"/f"{p.name}_gt_whole.txt").is_file())
        holdout = uavdt_holdout(videos, holdout_fraction)
        tracks = [t for name in holdout for t in uavdt_tracks(uavdt/"UAV-benchmark-M"/name,
                                                             uavdt/"UAV-benchmark-MOTD_v1.0/GT"/f"{name}_gt_whole.txt")]
        episodes += [("V3", e) for e in select_episodes(tracks, per_group=V3_PER_GROUP, max_frames=MAX_FRAMES,
                                                         max_per_video=8)]
    return episodes, holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--score", choices=["raw", "sigmoid"], default="sigmoid")
    parser.add_argument("--selection", choices=["upstream", "nan-safe"], default="nan-safe")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=["V1", "V2", "V3"], choices=["V1", "V2", "V3"])
    parser.add_argument("--track-grid", type=Path,
                        help="JSON list of {label, PENALTY_K/WINDOW_INFLUENCE/LR overrides}: one output/<label> per entry "
                             "(stage-3 post-processing grid; omitted = upstream cfg.TRACK, single output)")
    parser.add_argument("--uavdt-holdout-fraction", type=float, default=0.1,
                        help="must equal the trainer's --uavdt-holdout-fraction for V3 to be unseen")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from hift_research.checkpoint_tracker import build_tracker
    from hift_research.evaluation.validation import score_episode, summarize

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    episodes, holdout = build_episodes(args.data_root, args.subsets, args.uavdt_holdout_fraction)
    listing = [{"subset": s, "name": e["name"], "group": e["group"], "category": e["category"],
                "first_frame": str(Path(e["frames"][0]).name), "frames": len(e["frames"]), "scored": len(e["scored"])}
               for s, e in episodes]
    listing_sha = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
    grid = [{"label": None}] if args.track_grid is None else json.loads(args.track_grid.read_text(encoding="utf-8"))
    labels = [g["label"] for g in grid]
    if args.track_grid is not None and (len(set(labels)) != len(labels) or any(not l or "/" in l for l in labels)):
        raise ValueError("grid labels must be unique, non-empty and path-safe")
    read = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR)
    for entry in grid:
        overrides = {k: v for k, v in entry.items() if k != "label"}
        target = output if entry["label"] is None else output/entry["label"]
        target.mkdir(exist_ok=entry["label"] is None)
        tracker, info = build_tracker(ROOT, args.checkpoint, official=args.official, score=args.score,
                                      selection=args.selection, track_overrides=overrides)
        results, start_all = {}, time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for subset, episode in episodes:
                predictions, scores = [], []
                for index, image in enumerate(prefetched(pool, read, episode["frames"])):
                    if image is None:
                        raise ValueError(f"cannot decode {episode['frames'][index]}")
                    if index == 0:
                        tracker.init(image, episode["init_bbox"]); bbox = episode["init_bbox"]; best = float("nan")
                    else:
                        state = tracker.track(image); bbox = state["bbox"]; best = float(state.get("best_score", float("nan")))
                    predictions.append(list(map(float, bbox))); scores.append(best)
                results[(subset, episode["name"])] = score_episode(episode, predictions)
                safe = episode["name"].replace("/", "_").replace("#", "_")
                np.savetxt(target/f"{subset}__{safe}.csv", np.asarray(predictions), delimiter=",", fmt="%.10g")
                np.savetxt(target/f"{subset}__{safe}.score.csv", np.asarray(scores), fmt="%.6g")
        summary = summarize(results, [e for _, e in episodes])
        per_episode = {f"{s}/{n}": {"success_auc": m["success_auc"], "precision_at_20px": m["precision_at_20px"]}
                       for (s, n), m in results.items()}
        record = {"schema": "hift-domain-validation-v1", "tracker": info, "grid_label": entry["label"], "subsets": args.subsets,
                  "uavdt_holdout_fraction": args.uavdt_holdout_fraction, "uavdt_holdout_videos": holdout,
                  "episode_listing_sha256": listing_sha, "episodes": listing, "summary": summary, "per_episode": per_episode,
                  "seconds": time.perf_counter()-start_all, "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "software": {"torch": torch.__version__, "platform": platform.platform()},
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        with (target/"validation.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print("VALIDATION", entry["label"] or "", json.dumps({k: summary[k] for k in summary if k != "M_definition"}), flush=True)


if __name__ == "__main__":
    main()
