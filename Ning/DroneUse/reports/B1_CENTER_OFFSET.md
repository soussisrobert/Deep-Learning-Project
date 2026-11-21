# B1 candidate check: half-pixel centre offset

Status: diagnostic. The question comes from the deferred-item list ("half-pixel centre-offset"). It is not a B1
result, and it makes no benchmark or accuracy claim. Upstream, `src/**/inference.py`, `uav.py` and `assets.py` are
unchanged.

Files:

- Probe: `scripts/b1_center_offset_probe.py`
- Invariant tests: `tests/test_b1_center_offset.py`
- Runs: `artifacts/b1-center-offset/probe-002{-oracle,-b0,-ref}` and the combined summary `probe-002/summary.json`
  (sha256 `b5019983f110db5a…`). `probe-001` was stopped with no results; see Limitations. `smoke-*` are scratch runs.

## Question and preregistered rule (fixed before measuring)

Does the B0/REF inference pipeline put the predicted centre (or size) at a systematic sub-pixel offset because of
a coordinate-convention mismatch? A reviewer measured offline (not re-verified here) that validation predictions
sit +0.1 to +1.3 px right and +0.2 to +0.7 px down of GT. B0 and REF show the same V3 offset (dx +1.23 and +1.29),
and the offset does not correlate with velocity.

We call it a **pipeline bias** only if both of the following hold:

1. The mathematical audit predicts a specific offset.
2. The synthetic probe measures an offset of at least 0.25 px in that direction, with a 95% CI that excludes 0,
   for both B0 and REF and for every size.

Otherwise we conclude that no pipeline bias was found.

## Conventions

Pixel index i has its centre at i and covers the continuous interval [i, i+1). For a box (x, y, w, h), the index
centre is x+(w-1)/2 and the continuous centre is x+w/2. Every centre error below is
dx = (x_p+w_p/2) − (x_g+w_g/2). This difference is the same in either convention. It also equals the
evaluation's own convention (`evaluation/metrics.py:73`, which uses (w-1)/2 on both boxes).

## 1. Mathematical audit (code path used by `validate_domain.py`)

`validate_domain.py:113` calls `checkpoint_tracker.build_tracker`. That builds `tracking_nan_safe.make_nan_safe_tracker`
around `ScoreProbabilityAdapter` and either `load_original`, which loads the official checkpoint (sha256-pinned,
strict keys, `weights_only`), or the REF state with the corrected encoder layout. The NaN-safe subclass re-implements
`track()`, so our wrapper owns the search-crop call, the decode-to-image step, the state update and the output
conversion. Upstream owns `init`, `get_subwindow`, `generate_anchor` and the window.

| Id | Step | Convention (file:line) | Predicted effect on dx, dy |
|---|---|---|---|
| C1 | init state | centre = x+(w−1)/2 (index); size = (w, h) (`hift_tracker.py:81-85`) | reference |
| **G2** | crop origin | origin = floor(p−(S+1)/2+.5) = floor(p−S/2) (`base_tracker.py:51-56`). The crop centre is p+d, with d = floor(p−S/2)+(S−1)/2−p in (−1.5, −0.5], mean **−1.0**. This holds for the template (`hift_tracker.py:90,97-99`) and the search crop (`tracking_nan_safe.py:29`). | Decode adds offsets relative to p, so an exact locator lands **+1.0** px right/down (per frame in [+0.5, +1.5), phase-dependent). |
| C3 | resize | `cv2.resize` linear, centre-aligned: u = (x−origin+.5)·M/S−.5 (`base_tracker.py:88`) | none |
| C4 | crop scale | crop S = round(s_x); decode uses scale_z = 127/s_z unrounded (`tracking_nan_safe.py:27-29,57`) | zero-mean relative scale error of ±0.5/S |
| C5 | output grid | cell i at crop index 16i+63; cell 5 is at 143 = (287−1)/2. The training targets use the same grid (`hift_tracker.py:31-32`, `anchortarget.py:34`, `positive_targets.py:89`). | none: decode inverts the encoding (tested) |
| C6 | box decode | centre = g+(r−l)/2, w = l+r, d = 143·atanh(·), clipped at ±0.99 (`hift_tracker.py:22-39`) | none for \|d\| < 378 px |
| C7 | update | p ← p+(offset from 143)/scale_z; size is LR-smoothed (`tracking_nan_safe.py:57-65`) | carries G2 |
| C8 | window | 11×11 Hanning, symmetric about cell 5 (`hift_tracker.py:16-18`) | no directional bias |
| **G1** | output | x = cx−w/2 (`tracking_nan_safe.py:66`, `hift_tracker.py:182-185`) versus init (w−1)/2 | **−0.5** px left/up, exact |
| G3 | training label | `crop_like_SiamFC` centres x+w/2, the corners [x, y, x+w, y+h], as an index coordinate at crop index 255 (`source_ops.py:77-78`, `coco/par_crop.py:48-60`, `dataset.py:222`). Augmentation is index-consistent (`augmentation.py:119`). | Learned: a pure locator reports index c+0.5, i.e. +0.5 px (tested) |
| C10 | training crop scale | 510/side versus 127/s_z at inference | about −0.2% size |

Predictions (dx = dy):

| Locator | Real pipeline | Exact-crop arm |
|---|---|---|
| Exact locator (no learning) | G1+G2 = **+0.5** | G1 = **−0.5** |
| Network following the search appearance (G3 applies) | +1.0 | 0 |
| Network following the template centre | −0.5 on average | −0.5 |

So the audit predicts a specific offset only for the geometry. For a trained network the sign depends on how the
network uses the template, which the template-crop offset (also G2) shifts by the same amount as the search crop.

## 2. Synthetic probe (CPU, ground truth we control)

**Rendering.** Scenes are rendered analytically by area sampling: the background and a textured target are sums of
sinusoids integrated exactly per pixel, checked against 1000×1000 supersampling. Targets are square with sizes 12,
13, 24, 25, 48, 49, 96 and 97 px, at random sub-pixel phase in a 640² frame. No crop needed padding.

**Arms.** All arms use paired scenes.

- *pipeline*: the real tracker.
- *exact*: a probe-only analytic crop centred exactly on the state, which removes G2.

**Cases.**

- static: 20 seeds × 10 frames per size.
- moving: known sub-pixel velocity in [−1.2, 1.2] px/frame.
- whole-scene shift: 0.25, 0.5, 0.75, 1 and 2 px.

**Models.**

- B0: official checkpoint, sigmoid score, NaN-safe selection.
- REF: `TC2-P1-REF-001/final-model.pt` (sha256 `88fef3aa…`), same settings.
- Oracle: no network; the centroid of a coverage marker, in index convention.

**Statistics.** 95% CIs are percentile bootstraps over runs (5000 resamples).

### Predicted vs measured (static case, all sizes pooled; moving case within 0.1 px)

| Arm / component | Predicted | Oracle | B0 dx, dy | REF dx, dy |
|---|---|---|---|---|
| Exact crop (G1, plus the network) | −0.5 | −0.500, −0.500 | −0.86 [−0.95, −0.77], +0.02 [−0.08, 0.13] | −0.76 [−0.89, −0.64], −0.59 [−0.73, −0.46] |
| Paired pipeline − exact (G2 reaching the output) | +1.0 (locator) | +0.99 [0.98, 1.00], +0.99 | +0.27 [0.24, 0.31], +0.25 [0.21, 0.28] | +0.24 [0.20, 0.28], +0.39 [0.36, 0.42] |
| **Real pipeline** | +0.5 (locator) | +0.50 [0.49, 0.52], +0.49 | **−0.59 [−0.70, −0.49], +0.23 [0.12, 0.35]** | **−0.52 [−0.66, −0.39], −0.22 [−0.36, −0.08]** |
| Moving case, dx at zero velocity | none | n/a | −0.53 [−0.62, −0.44] (slope −0.04) | −0.47 [−0.59, −0.36] (slope −0.22) |
| Size dw, dh (pipeline) | −0.2% (C10) | −0.01, −0.01 | −0.74, +1.47 | −1.77, +4.85 |

Per size, real pipeline, static (* marks a CI that excludes 0):

| Model | Axis | 12 | 13 | 24 | 25 | 48 | 49 | 96 | 97 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | dx | −0.26* | −0.50* | −0.51* | −0.32* | −0.80* | −0.54* | −0.92* | −0.88* |
| B0 | dy | −0.16* | −0.21* | −0.11 | −0.03 | +0.18* | +0.32* | +0.96* | +0.90* |
| REF | dx | −0.20* | −0.34* | −0.34* | −0.23* | −0.68* | −0.46* | −0.93* | −0.94* |
| REF | dy | −0.08 | −0.17* | −0.02 | +0.16* | −0.30* | +0.01 | −0.55 | −0.80* |

**Shift equivariance.** e = c(δ) − c(0) − δ.

- Integer δ: e = 0 exactly in every arm and model.
- Fractional δ, real pipeline: mean |e| is 0.30–0.32 for the oracle, 0.33–0.51 for B0 and 0.31–0.52 for REF, with a
  maximum of 1.67 px. This is the per-frame jitter from integer crop origins.
- Fractional δ, exact crop: e = 0 for every model.

## Verdict (preregistered rule)

**No pipeline bias by the rule.** Condition 1 holds for the geometry: the audit predicts +0.5 px right/down, the
oracle confirms it, and the parts confirm it too (G1 −0.500, G2 +0.99). Condition 2 fails:

- **Direction.** Both networks are biased left in x on synthetic GT, the opposite of the predicted +0.5 and of the
  validation +x offset. In y the sign depends on size and model.
- **Magnitude.** In the static case, REF at sizes 12 and 25 is below 0.25 px. In the moving case, REF at size 97 has a
  CI that includes 0.

The two convention mismatches do exist (strict-xfail tests). Through the networks they nearly cancel:
G1 (−0.5) + G2 (+0.24…+0.39) ≈ −0.1 to −0.26 px. The rest is network-specific: the dy sign flips between B0 and REF
at 96 px, and dx grows with target size, from about −0.25 px at 12–13 px to −0.9 px at 96–97 px. The pipeline therefore cannot
produce the observed +1.2 px V3 offset. That offset is consistent with an annotation-convention difference (the
reviewer's hypothesis), or with a network response specific to real content. Both B0 and REF would share either
cause. This report does not prove the annotation cause.

**No B1 candidate is proposed.** If one is ever pursued, it must be preregistered separately. The training-consistent
geometry would be: init centre x+w/2 (the G3 label point), a sub-pixel-exact centred crop via an affine warp in a
tracker subclass (an adapter, not an edit to protected files), and output kept as cx−w/2. It would be validated once,
first on this probe and then on V, never tuned on V.

## Limitations

- Synthetic scenes only: sinusoid textures, square targets, no blur, noise, JPEG, rotation, occlusion or distractors.
  The network-specific part of the offset depends on content, so synthetic magnitudes do not transfer to real data.
- The probe ran on CPU in float32 with `.cuda()` mapped to identity in the probe process. The checkpoint rules of
  `build_tracker` were unchanged. Results are close to, but not bitwise equal to, the GPU results.
- The exact arm is a counterfactual that needs the analytic scene. It isolates G2 but keeps the inference template
  convention (index centre), not the training one (G3), so the purely intrinsic network bias is not measured.
- The validation offsets were not re-measured here. Per-dataset annotation conventions (V1–V3) were not audited.
- `probe-001` was stopped: thread oversubscription stalled it while other jobs loaded this shared machine. It wrote
  no results. `probe-002` ran with one torch thread and one BLAS thread per model.
- The run was at commit 8e3ca6c (dirty: another agent's `reports/TC2_STATE.md` plus these untracked files). The
  dirty-diff sha256 is `7a6bf61c…` and the upstream is 992aa7c.

## Reproduce

```
python scripts/b1_center_offset_probe.py --models b0 --seeds 20 --frames 10 --shift-seeds 2 --threads 1 --run-id <new>
python scripts/b1_center_offset_probe.py --combine <oracle-id> <b0-id> <ref-id> --run-id <new>
python -m pytest tests/test_b1_center_offset.py -q
```
