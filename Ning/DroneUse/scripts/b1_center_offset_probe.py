"""B1 half-pixel centre-offset probe: synthetic ground truth we control, CPU only.

Question: does the B0/REF inference pipeline put the predicted box centre (or size) at a systematic sub-pixel
offset because of a coordinate-convention mismatch? (reports/B1_CENTER_OFFSET.md has the audit and verdict.)

Scenes are rendered ANALYTICALLY with an area-sampling camera: pixel i covers the continuous interval [i, i+1),
the background and the target texture are sums of sinusoids integrated exactly over each pixel (or over the
part of a pixel the target rectangle covers), so a target at any sub-pixel position, and any whole-scene
translation, is rendered exactly. GT boxes are xywh with continuous extents; the reported centre error is
dx = (x_p + w_p/2) - (x_g + w_g/2), which equals the evaluation's (w-1)/2 convention difference
(src/hift_research/evaluation/metrics.py:73) because the -1/2 terms cancel.

Arms (same scenes for every model: paired design):
  pipeline  the real NaN-safe tracker (build_tracker, i.e. validate_domain's tracker) with upstream get_subwindow.
  exact     probe-only counterfactual: get_subwindow replaced by an analytic crop centred EXACTLY on the tracker's
            index-convention state (continuous pos+0.5), no integer-origin quantisation. Not a B1 correction: it
            needs the analytic scene and exists only to separate crop geometry from the network.
Models: b0 (official checkpoint, sigmoid score, NaN-safe), ref (TC2-P1-REF-001 final-model.pt, same settings),
        oracle (no network: index-convention centroid of a coverage marker in the crop; isolates geometry).
Cases: static (P1), moving with known sub-pixel velocity (P2), whole-scene translation by integer and
       fractional offsets (P3: shift equivariance).

CPU: upstream ModelBuilder hardcodes .cuda(); this probe maps Module.cuda/Tensor.cuda to identity in its own
process and replaces inference.load_original with the same body minus the CUDA requirement (disclosed in the
manifest). build_tracker's checkpoint rules are used unchanged (official sha256 pin, strict keys, weights_only).
Diagnostic only: no benchmark claim. Output: artifacts/b1-center-offset/<run-id>/ (never overwritten).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if __name__ == "__main__":  # before numpy/torch load: CPU only, single-threaded BLAS (torch pool set in main)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _name in THREAD_ENV:
        os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF_CHECKPOINT = ROOT/"artifacts/tc2-runs/TC2-P1-REF-001/final-model.pt"
SIZES = (12, 13, 24, 25, 48, 49, 96, 97)
FRAME = 640
GRID = 16*np.arange(11)+63.0  # crop-index positions of the 11x11 output cells (hift_tracker.py:31-32)
SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)


# ----------------------------------------------------------------------------------------------- scene
@dataclass(frozen=True)
class Field:
    """mean + sum_k Re(weight_k,c * exp(i 2 pi f_k . X)); weight = amplitude * exp(i phase)."""
    mean: np.ndarray    # (3,)
    freq: np.ndarray    # (K, 2) cycles per pixel
    weight: np.ndarray  # (K, 3) complex


@dataclass(frozen=True)
class Scene:
    background: Field
    texture: Field              # in target coordinates (moves with the box)
    box: tuple                  # (x0, y0, w, h) continuous image coordinates
    shift: tuple = (0.0, 0.0)   # whole-world translation of the background
    marker: bool = False        # oracle scenes: channel 2 = 255 * target coverage, background channel 2 = 0


def axis_integrals(freq, a, b):
    """(K, N) complex: integral over [a_n, b_n] of exp(i 2 pi f_k x) dx (exact; np.sinc is normalised)."""
    length = b-a
    return length[None, :]*np.exp(1j*np.pi*freq[:, None]*(a+b)[None, :])*np.sinc(freq[:, None]*length[None, :])


def field_integral(field, xa, xb, ya, yb):
    """(H, W, 3): exact integral of the field over every rectangle [xa_j, xb_j] x [ya_i, yb_i]."""
    ix = axis_integrals(field.freq[:, 0], xa, xb)
    iy = axis_integrals(field.freq[:, 1], ya, yb)
    area = np.outer(yb-ya, xb-xa)
    out = np.empty((len(ya), len(xa), 3))
    for c in range(3):
        out[..., c] = field.mean[c]*area+np.real((iy.T*field.weight[:, c]) @ ix)
    return out


def pixel_edges(origin, pitch, count):
    lo = origin+pitch*np.arange(count)
    return lo, lo+pitch


def render_background(scene, ox, oy, pitch, height, width):
    xa, xb = pixel_edges(ox, pitch, width)
    ya, yb = pixel_edges(oy, pitch, height)
    sx, sy = scene.shift
    out = field_integral(scene.background, xa-sx, xb-sx, ya-sy, yb-sy)/pitch**2
    if scene.marker:
        out[..., 2] = 0.0
    return out


def render(scene, ox, oy, pitch, height, width, base=None):
    """Area-sampled image of the scene on the grid whose pixel (i, j) covers
    [ox + j*pitch, ox + (j+1)*pitch] x [oy + i*pitch, oy + (i+1)*pitch]. Returns float (H, W, 3)."""
    out = render_background(scene, ox, oy, pitch, height, width) if base is None else base.copy()
    xa, xb = pixel_edges(ox, pitch, width)
    ya, yb = pixel_edges(oy, pitch, height)
    x0, y0, w, h = scene.box
    la, lb = np.maximum(xa, x0), np.minimum(xb, x0+w)
    ta, tb = np.maximum(ya, y0), np.minimum(yb, y0+h)
    cols, rows = np.flatnonzero(lb > la), np.flatnonzero(tb > ta)
    if not len(cols) or not len(rows):
        return out
    la, lb, ta, tb = la[cols], lb[cols], ta[rows], tb[rows]
    sx, sy = scene.shift
    covered_bg = field_integral(scene.background, la-sx, lb-sx, ta-sy, tb-sy)
    covered_tex = field_integral(scene.texture, la-x0, lb-x0, ta-y0, tb-y0)
    block = np.ix_(rows, cols)
    out[block] += (covered_tex-covered_bg)/pitch**2
    if scene.marker:
        out[..., 2][block] = 255.0*np.outer(tb-ta, lb-la)/pitch**2
    return out


def to_uint8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def random_field(rng, count, fmin, fmax, std, mean):
    magnitude = np.exp(rng.uniform(np.log(fmin), np.log(fmax), count))
    angle = rng.uniform(0, 2*np.pi, count)
    freq = np.stack([magnitude*np.cos(angle), magnitude*np.sin(angle)], 1)
    amplitude = rng.uniform(0.5, 1.5, (count, 3))*std/np.sqrt(count*0.5*(13/12))  # E[u^2] = 13/12, u~U(.5,1.5)
    weight = amplitude*np.exp(1j*rng.uniform(0, 2*np.pi, (count, 3)))
    return Field(np.asarray(mean, float), freq, weight)


def make_scene(case, size, seed, marker=False):
    """Deterministic scene per (case, size, seed); marker does not change any random draw."""
    rng = np.random.default_rng([{"static": 1, "moving": 2, "shift": 3, "test": 9}[case], size, seed])
    background_mean = np.array([110.0, 115.0, 105.0])
    background = random_field(rng, 48, 1/64, 1/3, 30.0, background_mean)
    while True:
        target_mean = rng.uniform(30, 225, 3)
        if np.linalg.norm(target_mean-background_mean) > 80:
            break
    texture = random_field(rng, 10, 1/size, min(1/3, 4/size), 35.0, target_mean)
    x0 = FRAME/2-size/2+rng.uniform(-6, 6)
    y0 = FRAME/2-size/2+rng.uniform(-6, 6)
    velocity = rng.uniform(-1.2, 1.2, 2)
    return Scene(background, texture, (x0, y0, float(size), float(size)), marker=marker), velocity


def sequence(case, size, seed, frames, marker=False, shift=0.0):
    """Scenes for frames 0..frames (frame 0 = initialisation)."""
    scene, velocity = make_scene(case, size, seed, marker)
    x0, y0, w, h = scene.box
    if case == "shift":
        scene = replace(scene, shift=(shift, shift), box=(x0+shift, y0+shift, w, h))
        return [scene]*(frames+1), np.zeros(2)
    if case == "static":
        return [scene]*(frames+1), np.zeros(2)
    return [replace(scene, box=(x0+velocity[0]*t, y0+velocity[1]*t, w, h)) for t in range(frames+1)], velocity


# ---------------------------------------------------------------------------------------------- oracle
class OracleLocator:
    """No network: index-convention centroid (u, v) and extent of the coverage marker (channel 2) in the crop,
    emitted at every cell with the training label encoding tanh(d/143) (anchortarget.py:94-98), so every cell
    decodes to the same box. This is an unbiased locator of the target's pixel-index centre in crop pixels."""
    def eval(self):
        return self

    def template(self, z):
        return None

    def track(self, x):
        import torch
        marker = x[0, 2].double().numpy()/255.0
        cols, rows = marker.sum(0), marker.sum(1)
        mass = marker.sum()
        if mass <= 0:
            raise ValueError("oracle: no target coverage in the crop")
        u = float((cols*np.arange(len(cols))).sum()/mass)
        v = float((rows*np.arange(len(rows))).sum()/mass)
        w, h = float(rows.max()), float(cols.max())  # interior row sum = width, interior column sum = height
        gx, gy = np.meshgrid(GRID, GRID)  # [row=yy, col=xx]
        distances = np.stack([gx-(u-w/2), (u+w/2)-gx, gy-(v-h/2), (v+h/2)-gy])
        loc = torch.from_numpy(np.tanh(distances/143.0)[None])
        cls1 = torch.zeros(1, 2, 11, 11, dtype=torch.float64)
        cls1[0, 1, int(np.clip(np.rint((v-63)/16), 0, 10)), int(np.clip(np.rint((u-63)/16), 0, 10))] = 4.0
        cls2 = torch.full((1, 1, 11, 11), -2.0, dtype=torch.float64)
        return {"loc": loc, "cls1": cls1, "cls2": cls2}


# -------------------------------------------------------------------------------------------- trackers
def enable_cpu_upstream():
    """Probe-process-only CPU mapping of upstream's hardcoded .cuda() (model_builder.py:22-23, utile.py:96-97,
    base_tracker.py get_subwindow) and a CPU twin of inference.load_original (same strict loading)."""
    import torch
    import hift_research.inference as inference
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self

    def load_original_cpu(root, checkpoint=None, config=None):
        inference.configure_upstream(root, config)
        from pysot.models.model_builder import ModelBuilder
        model = ModelBuilder().eval()
        if checkpoint is not None:
            model.load_state_dict(inference.read_checkpoint_state(checkpoint), strict=True)
        return model
    inference.load_original = load_original_cpu
    return "torch.nn.Module.cuda/torch.Tensor.cuda -> identity; inference.load_original without CUDA check"


def build(kind):
    """(tracker, info) for 'b0', 'ref' (validate_domain's build_tracker, sigmoid + nan-safe) or 'oracle'."""
    from hift_research.checkpoint_tracker import build_tracker
    from hift_research.inference import ScoreProbabilityAdapter, configure_upstream
    if kind == "oracle":
        configure_upstream(ROOT)
        from hift_research.tracking_nan_safe import make_nan_safe_tracker
        tracker, info = make_nan_safe_tracker(ScoreProbabilityAdapter(OracleLocator())), {"model": "oracle"}
    elif kind == "b0":
        tracker, info = build_tracker(ROOT, official=True, score="sigmoid", selection="nan-safe")
    elif kind == "ref":
        tracker, info = build_tracker(ROOT, REF_CHECKPOINT, score="sigmoid", selection="nan-safe")
    else:
        raise ValueError(kind)
    from pysot.core.config import cfg
    cfg.CUDA = False
    return tracker, info


def exact_variant(tracker):
    """Copy of the tracker (shared model) whose crops are rendered analytically, centred exactly on the state."""
    import torch

    class ExactCropTracker(type(tracker)):
        def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
            pitch = original_sz/model_sz
            patch = to_uint8(render(self.scene, pos[0]+0.5-original_sz/2, pos[1]+0.5-original_sz/2,
                                    pitch, model_sz, model_sz))
            return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)[None]).astype(np.float32))
    clone = copy.copy(tracker)
    clone.__class__ = ExactCropTracker
    return clone


def crop_inside(center, source_size, shape):
    from hift_research.geometry import CropGeometry
    geometry = CropGeometry.from_tracker(center, int(source_size), 287)
    return (geometry.origin_x >= 0 and geometry.origin_y >= 0 and geometry.origin_x+source_size <= shape[1]
            and geometry.origin_y+source_size <= shape[0])


def track(tracker, scenes, exact):
    """Initialise on scenes[0].box (exact GT), track the rest. Returns predictions (N, 4) and padded-crop count."""
    from pysot.core.config import cfg
    dummy = np.zeros((FRAME, FRAME, 3), np.uint8)
    base = None if exact else render_background(scenes[0], 0.0, 0.0, 1.0, FRAME, FRAME)
    predictions, padded = [], 0
    for t, scene in enumerate(scenes):
        tracker.scene = scene
        image = dummy if exact else to_uint8(render(scene, 0.0, 0.0, 1.0, FRAME, FRAME, base=base))
        if t == 0:
            tracker.init(image, list(scene.box))
            continue
        size = tracker.size
        s_z = np.sqrt((size[0]+cfg.TRACK.CONTEXT_AMOUNT*size.sum())*(size[1]+cfg.TRACK.CONTEXT_AMOUNT*size.sum()))
        padded += not crop_inside(tracker.center_pos, round(s_z*cfg.TRACK.INSTANCE_SIZE/cfg.TRACK.EXEMPLAR_SIZE),
                                  (FRAME, FRAME))
        predictions.append([float(v) for v in tracker.track(image)["bbox"]])
    return np.asarray(predictions), padded


def errors(predictions, scenes):
    gt = np.asarray([s.box for s in scenes[1:]], float)
    centre = (predictions[:, :2]+predictions[:, 2:]/2)-(gt[:, :2]+gt[:, 2:]/2)
    return {"dx": centre[:, 0], "dy": centre[:, 1], "dw": predictions[:, 2]-gt[:, 2], "dh": predictions[:, 3]-gt[:, 3]}


# ------------------------------------------------------------------------------------------ statistics
def bootstrap_mean(values, seed=0, resamples=5000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def bootstrap_intercept(x, y, seed=0, resamples=5000):
    """Least-squares intercept of y = a + b x (bias at zero velocity) with a run-level bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = np.stack([np.ones_like(x), x], 1)
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        pick = rng.integers(0, len(x), len(x))
        draws.append(np.linalg.lstsq(design[pick], y[pick], rcond=None)[0])
    draws = np.asarray(draws)
    return {"n": int(len(x)), "intercept": float(fit[0]), "slope": float(fit[1]),
            "intercept_ci95": [float(np.percentile(draws[:, 0], 2.5)), float(np.percentile(draws[:, 0], 97.5))]}


def summarize(rows):
    """rows: per-run records. Returns nested summary model -> arm -> case -> size|'all' -> metric -> stats."""
    summary = {}
    keys = sorted({(r["model"], r["arm"], r["case"]) for r in rows if r["case"] != "shift"})
    for model, arm, case in keys:
        mine = [r for r in rows if (r["model"], r["arm"], r["case"]) == (model, arm, case)]
        block = {}
        for size in sorted({r["size"] for r in mine})+["all"]:
            sel = [r for r in mine if size == "all" or r["size"] == size]
            block[str(size)] = {m: bootstrap_mean([r[m] for r in sel]) for m in ("dx", "dy", "dw", "dh")}
            if case == "moving":
                block[str(size)]["dx_at_zero_velocity"] = bootstrap_intercept([r["vx"] for r in sel], [r["dx"] for r in sel])
                block[str(size)]["dy_at_zero_velocity"] = bootstrap_intercept([r["vy"] for r in sel], [r["dy"] for r in sel])
        summary.setdefault(model, {}).setdefault(arm, {})[case] = block
    return summary


def equivariance(shift_rows):
    """Per model/arm: e_t = c_t(delta) - c_t(0) - delta (both axes) for integer and fractional delta."""
    out = {}
    index = {(r["model"], r["arm"], r["size"], r["seed"], r["shift"]): r for r in shift_rows}
    for (model, arm, size, seed, shift), row in index.items():
        if shift == 0.0:
            continue
        base = index[(model, arm, size, seed, 0.0)]
        e = np.concatenate([np.asarray(row["cx"])-np.asarray(base["cx"])-shift,
                            np.asarray(row["cy"])-np.asarray(base["cy"])-shift])
        out.setdefault(model, {}).setdefault(arm, {}).setdefault(str(shift), []).append(e)
    result = {}
    for model, arms in out.items():
        for arm, shifts in arms.items():
            for shift, chunks in shifts.items():
                e = np.concatenate(chunks)
                result.setdefault(model, {}).setdefault(arm, {})[shift] = {
                    "frames_x_axes": int(len(e)), "mean": float(e.mean()), "mean_abs": float(np.abs(e).mean()),
                    "max_abs": float(np.abs(e).max()), "fraction_abs_gt_0.25": float((np.abs(e) > 0.25).mean())}
    phase = {}
    for (model, arm, size, seed, shift), row in index.items():
        phase.setdefault(model, {}).setdefault(arm, {}).setdefault(str(shift), []).append(
            (np.mean(row["dx"])+np.mean(row["dy"]))/2)
    for model in phase:
        for arm in phase[model]:
            phase[model][arm] = {s: bootstrap_mean(v) for s, v in phase[model][arm].items()}
    return {"equivariance_error": result, "bias_by_world_shift": phase}


def decision(summary, threshold=0.25):
    """Preregistered measured half of the rule: for BOTH b0 and ref, pipeline arm, every size, mean offset
    >= threshold in one common direction with a 95% CI excluding 0 (checked per axis and per case)."""
    verdict = {}
    for case in ("static", "moving"):
        for axis in ("dx", "dy"):
            stats = [summary.get(m, {}).get("pipeline", {}).get(case, {}).get(str(s), {}).get(axis)
                     for m in ("b0", "ref") for s in SIZES]
            if any(s is None for s in stats):
                verdict[f"{case}/{axis}"] = "incomplete"
                continue
            signs = {np.sign(s["mean"]) for s in stats}
            ok = (len(signs) == 1 and all(abs(s["mean"]) >= threshold for s in stats)
                  and all(s["ci95"][0] > 0 or s["ci95"][1] < 0 for s in stats))
            verdict[f"{case}/{axis}"] = {"criterion_met": bool(ok),
                                         "min_abs_mean": float(min(abs(s["mean"]) for s in stats)),
                                         "means": [round(s["mean"], 4) for s in stats]}
    return verdict


# ------------------------------------------------------------------------------------------------ main
def run_model(kind, arms, seeds, frames, shift_seeds, sizes, log):
    tracker, info = build(kind)
    trackers = {"pipeline": tracker, "exact": exact_variant(tracker)}
    marker = kind == "oracle"
    rows = []
    for arm in arms:
        for case in ("static", "moving"):
            for size in sizes:
                for seed in range(seeds):
                    scenes, velocity = sequence(case, size, seed, frames, marker)
                    predictions, padded = track(trackers[arm], scenes, arm == "exact")
                    e = errors(predictions, scenes)
                    rows.append({"model": kind, "arm": arm, "case": case, "size": size, "seed": seed,
                                 "vx": float(velocity[0]), "vy": float(velocity[1]), "padded_crops": padded,
                                 **{k: float(v.mean()) for k, v in e.items()},
                                 "per_frame": {k: [round(float(x), 5) for x in v] for k, v in e.items()}})
                log(f"{kind} {arm} {case} size {size}")
            log(f"{kind} {arm} {case} done")
        for size in sizes:
            for seed in range(shift_seeds):
                for shift in SHIFTS:
                    scenes, _ = sequence("shift", size, seed, max(2, frames//2), marker, shift)
                    predictions, padded = track(trackers[arm], scenes, arm == "exact")
                    e = errors(predictions, scenes)
                    centre = predictions[:, :2]+predictions[:, 2:]/2
                    rows.append({"model": kind, "arm": arm, "case": "shift", "size": size, "seed": seed,
                                 "shift": shift, "padded_crops": padded, "cx": centre[:, 0].tolist(),
                                 "cy": centre[:, 1].tolist(), "dx": e["dx"].tolist(), "dy": e["dy"].tolist()})
        log(f"{kind} {arm} shift done")
    counters = {k: int(getattr(type(tracker), k, 0)) for k in ("frames", "nonfinite_cells", "first_nan_would_have_won")}
    return rows, {**info, "nan_safe_counters_pipeline": counters}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["oracle", "b0", "ref"], choices=["oracle", "b0", "ref"])
    parser.add_argument("--arms", nargs="+", default=["pipeline", "exact"], choices=["pipeline", "exact"])
    parser.add_argument("--seeds", type=int, default=10, help="runs per size for the static and moving cases")
    parser.add_argument("--shift-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--combine", nargs="+", metavar="RUN_ID",
                        help="no tracking: summarise the runs.jsonl of these finished runs (e.g. one per model)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/"src"))
    import cv2
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    output = ROOT/"artifacts/b1-center-offset"/args.run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    console = (output/"console.txt").open("x", encoding="utf-8")

    def log(message):
        line = f"[{time.perf_counter()-started:7.1f}s] {message}"
        print(line, flush=True)
        console.write(line+"\n")
        console.flush()
    from hift_research.provenance import sha256, source_identity
    identity = source_identity(ROOT)
    rows, infos, parts = [], {}, None
    if args.combine:
        cpu_patch, parts = "n/a (combine only; see parts)", {}
        for run_id in args.combine:
            part = ROOT/"artifacts/b1-center-offset"/run_id
            record = json.loads((part/"summary.json").read_text(encoding="utf-8"))
            parts[run_id] = {"runs_jsonl_sha256": sha256(part/"runs.jsonl"), "summary_sha256": sha256(part/"summary.json"),
                             "settings": record["settings"], "software": record["software"],
                             "git_commit": record["git_commit"], "dirty_diff_sha256": record["dirty_diff_sha256"],
                             "cpu_adapter": record["cpu_adapter"], "seconds": record["seconds"]}
            infos.update(record["trackers"])
            rows += [json.loads(line) for line in (part/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
        if len({(r["model"], r["arm"], r["case"], r["size"], r["seed"], r.get("shift")) for r in rows}) != len(rows):
            raise ValueError("combined runs overlap")
    else:
        cpu_patch = enable_cpu_upstream()
        for kind in args.models:
            model_rows, infos[kind] = run_model(kind, args.arms, args.seeds, args.frames, args.shift_seeds,
                                                args.sizes, log)
            rows += model_rows
        with (output/"runs.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row)+"\n")
    summary = summarize([r for r in rows if r["case"] != "shift"])
    shift = equivariance([r for r in rows if r["case"] == "shift"])
    record = {
        "schema": "hift-b1-center-offset-probe-v1",
        "identity": "diagnostic (synthetic ground truth); not B0/B1/B2 results, no benchmark claim",
        "git_commit": identity["git_commit"], "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "git_status": identity["git_status"], "source_tree_sha256": identity["source_tree_sha256"],
        "upstream_commit": identity["upstream_commit"],
        "config_sha256": sha256(ROOT/"third_party/HiFT/experiments/config.yaml"),
        "trackers": infos, "cpu_adapter": cpu_patch, "combined_parts": parts,
        "settings": {"combined_run_ids": args.combine} if args.combine else {
            "models": args.models, "arms": args.arms, "seeds": args.seeds, "shift_seeds": args.shift_seeds,
            "frames": args.frames, "sizes": args.sizes, "frame_px": FRAME, "shifts": SHIFTS,
            "scene_seed": "np.random.default_rng([case_id, size, seed])", "bootstrap": "5000, seed 0",
            "renderer": "analytic area sampling (sum of sinusoids), uint8 rounding"},
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor(),
                     "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "thread_env": {k: os.environ.get(k) for k in THREAD_ENV}},
        "padded_crops_total": int(sum(r["padded_crops"] for r in rows)),
        "summary": summary, "shift": shift, "decision_measured_half": decision(summary),
        "seconds": time.perf_counter()-started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["synthetic textures/backgrounds only; square targets; no blur, noise, JPEG or rotation",
                        "CPU float32 instead of CUDA; numerically close but not bitwise the GPU results",
                        "the exact arm is a probe-only counterfactual crop, not a deployable correction"],
    }
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    log(f"wrote {output}")
    for model, arms in summary.items():
        for arm, cases in arms.items():
            for case, block in cases.items():
                a = block["all"]
                log(f"{model:6s} {arm:8s} {case:7s} dx {a['dx']['mean']:+.3f} {a['dx']['ci95']} "
                    f"dy {a['dy']['mean']:+.3f} {a['dy']['ci95']} dw {a['dw']['mean']:+.3f} dh {a['dh']['mean']:+.3f}")
    log(json.dumps(record["decision_measured_half"]))
    console.close()


if __name__ == "__main__":
    main()
