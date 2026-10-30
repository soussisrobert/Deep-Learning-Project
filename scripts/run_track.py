"""Track a benchmark with HiFT under one condition and save metrics.

A condition = model variant x attack x defence. Examples:
  python scripts/run_track.py --dataset dtb70 --root data/DTB70 --snapshot weights/general_model.pth
  python scripts/run_track.py ... --attack pgd --eps 4 --steps 10
  python scripts/run_track.py ... --attack pgd --eps 4 --defense blur1 --adaptive
  python scripts/run_track.py ... --variant int8bb --device cpu
Results go to results/<run-name>/ : summary.json, per_seq.json, preds/<seq>.txt (pysot format).
"""
import argparse
import json
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)  # torch quantisation deprecation notices
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiftstress.attacks import make_attack  # noqa: E402
from hiftstress.crops import gt_crops, gt_pairs  # noqa: E402
from hiftstress.data import load_dataset  # noqa: E402
from hiftstress.defenses import make_defense  # noqa: E402
from hiftstress.edge import make_variant  # noqa: E402
from hiftstress.metrics import aggregate, sequence_scores  # noqa: E402
from hiftstress.model import build_model, pick_device  # noqa: E402
from hiftstress.tracker import TrackingModel, run_sequence  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "dtb70", "flat"])
    p.add_argument("--root", help="dataset folder")
    p.add_argument("--max-seqs", type=int, help="spread subset of N sequences (quick runs)")
    p.add_argument("--seqs", nargs="*", help="exact sequence names")
    p.add_argument("--seqs-file", help="JSON with a 'sequences' list (e.g. attack-subset-40.json) or one name per line")
    p.add_argument("--max-frames", type=int, help="truncate every sequence (quick runs only)")
    p.add_argument("--snapshot", help="HiFT weights (general_model.pth)")
    p.add_argument("--allow-random", action="store_true", help="run without weights (pipeline test)")
    p.add_argument("--device", default="auto")
    p.add_argument("--variant", default="fp32")
    p.add_argument("--calib-seqs", type=int, default=20, help="sequences used to calibrate int8 variants")
    p.add_argument("--calib-overlap-ok", action="store_true",
                   help="allow calibration sequences that are also evaluated (default: excluded)")
    p.add_argument("--attack-device", default=None,
                   help="device for the attacker's fp32 network (e.g. cuda while an int8 model tracks on cpu)")
    p.add_argument("--attack", default="none", choices=["none", "noise", "fgsm", "pgd"])
    p.add_argument("--eps", type=float, default=0, help="L-inf budget in pixels (of 255)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--adaptive", action="store_true", help="attack differentiates through the defence")
    p.add_argument("--defense", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    p.add_argument("--out", default="results")
    p.add_argument("--name", help="run name (default built from the condition)")
    return p.parse_args()


def main():
    a = parse()
    if not a.snapshot and not a.allow_random:
        sys.exit("Give --snapshot weights/general_model.pth, or --allow-random for a pipeline test.")
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    if a.seqs_file:
        txt = Path(a.seqs_file).read_text()
        a.seqs = json.loads(txt)["sequences"] if a.seqs_file.endswith(".json") else txt.split()
    seqs = load_dataset(a.dataset, a.root, a.max_seqs, a.seqs, a.max_frames, a.seed)

    net_fp32, model_info = build_model(a.snapshot, device)
    calib, pairs, calib_names = None, None, []
    if a.variant in ("int8bb", "int8full"):
        pool = load_dataset(a.dataset, a.root, None, None, None, a.seed + 1)
        if not a.calib_overlap_ok:
            evald = {s.name for s in seqs}
            pool = [s for s in pool if s.name not in evald] or pool
        idx = np.linspace(0, len(pool) - 1, num=min(a.calib_seqs, len(pool))).round().astype(int)
        cal_seqs = [pool[i] for i in sorted(set(idx))]
        calib_names = [s.name for s in cal_seqs]
        _, calib = gt_crops(cal_seqs, frames_per_seq=2, device="cpu")
        pairs = gt_pairs(cal_seqs, frames_per_seq=2, device="cpu")
    net, dtype, run_device, var_info = make_variant(a.variant, net_fp32, device, calib, pairs)
    if calib_names:
        var_info["calibration_sequences"] = calib_names
    if a.attack_device:
        attack_net = net_fp32.to(torch.device(a.attack_device))
    else:
        attack_net = net_fp32 if run_device == device else net_fp32.to(run_device)

    attack = make_attack(a.attack, a.eps, a.steps, a.adaptive, a.seed)
    defense = make_defense(a.defense)
    cond = f"{a.variant}_{a.attack if attack else 'clean'}{('%g' % a.eps) if attack else ''}" \
           f"{'s%d' % a.steps if a.attack == 'pgd' else ''}_{defense.name if defense else 'nodef'}" \
           f"{'_adaptive' if a.adaptive and attack and defense else ''}"
    name = a.name or f"{a.dataset}{'-sub%d' % a.max_seqs if a.max_seqs else ''}_{cond}"
    out = Path(a.out) / name
    (out / "preds").mkdir(parents=True, exist_ok=True)
    print(f"[run] {name} on {run_device} | {len(seqs)} sequences | {model_info.get('warning', 'weights ' + str(a.snapshot))}")

    per_seq, totals = [], {"attack_s": 0.0, "defense_s": 0.0, "model_s": 0.0, "frames": 0, "wall_s": 0.0}
    drops = []
    t_all = time.perf_counter()
    for k, seq in enumerate(seqs):
        model = TrackingModel(net, run_device, dtype, attack, defense, attack_net)
        preds, scores, times = run_sequence(seq, model, run_device)
        np.savetxt(out / "preds" / f"{seq.name}.txt", preds, fmt="%.4f", delimiter=",")
        s = sequence_scores(preds, seq.gt)
        s.update(name=seq.name, fps=float((len(times) - 1) / max(times[1:].sum(), 1e-9)),
                 mean_score=float(np.nanmean(scores)) if len(scores) > 1 else float("nan"))
        per_seq.append(s)
        for key in ("attack_s", "defense_s", "model_s", "frames"):
            totals[key] += model.stats[key]
        totals["wall_s"] += float(times[1:].sum())
        if attack is not None and hasattr(attack, "last_drop"):
            drops += attack.last_drop
            attack.last_drop = []
        print(f"  ({k + 1:3d}/{len(seqs)}) {seq.name:24s} AUC {s['success_auc']:.3f}  "
              f"P@20 {s['precision_20']:.3f}  {s['fps']:6.1f} fps")

    summary = aggregate(per_seq)
    n = max(totals["frames"], 1)
    summary.update(
        run=name, dataset=a.dataset, root=a.root, variant=a.variant, attack=a.attack if attack else "none",
        eps=a.eps if attack else 0, steps=a.steps if a.attack == "pgd" else (1 if a.attack == "fgsm" else 0),
        adaptive=bool(a.adaptive and attack and defense), defense=defense.name if defense else "none",
        device=str(run_device), attack_device=str(next(attack_net.parameters()).device) if attack else None,
        seqs_file=a.seqs_file, torch=torch.__version__, seed=a.seed, model=model_info, variant_info=var_info,
        fps_end_to_end=totals["frames"] / max(totals["wall_s"], 1e-9),
        ms_per_frame={"model": 1e3 * totals["model_s"] / n, "attack": 1e3 * totals["attack_s"] / n,
                      "defense": 1e3 * totals["defense_s"] / n, "total": 1e3 * totals["wall_s"] / n},
        total_minutes=(time.perf_counter() - t_all) / 60,
    )
    if drops:
        d = np.array(drops)
        summary["attack_objective"] = {"clean_mean": float(d[:, 0].mean()), "attacked_mean": float(d[:, 1].mean()),
                                       "frames_reduced": float((d[:, 1] < d[:, 0]).mean())}
    summary["success_curve"] = [round(v, 5) for v in summary["success_curve"]]
    summary["precision_curve"] = [round(v, 5) for v in summary["precision_curve"]]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_seq.json").write_text(json.dumps(per_seq, indent=1))
    print(f"[done] {name}: success AUC {summary['success_auc']:.3f} | precision@20 {summary['precision_20']:.3f} | "
          f"{summary['fps_end_to_end']:.1f} fps end-to-end | {summary['total_minutes']:.1f} min")


if __name__ == "__main__":
    main()
