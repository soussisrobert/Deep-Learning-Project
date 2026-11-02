# Stage 3 preregistration — tracker post-processing on frozen weights, 2026-11-02

Written and committed before any stage-3 grid result exists.

## Why (measured, stage-2 data)
- Five training-recipe variants never beat REF on V. Per-episode AUC correlates 0.91-0.95 across all runs,
  including B0: the same episodes fail for every model trained this way (Opus review, 2026-11-02).
- The inference change raw -> sigmoid score (+ NaN-safe argmax) moved B0 by +0.044 M on V1+V2, more than any
  training change. The upstream post-processing constants (config.yaml: PENALTY_K 0.08, WINDOW_INFLUENCE 0.42,
  LR 0.30) were set for the raw score (B0 diagnostics: selected score median 2.35 raw vs 0.97 sigmoid). Under the
  sigmoid score the cosine window weighs relatively more and the size update `lr = penalty*score*LR` is slower
  (hift_tracker.py:158-164).
- Per-frame diagnosis of the saved REF predictions (episode listing hash verified against the TC2 runs):
  abrupt loss (IoU < 0.05 for >= 10 scored frames) explains ~0.40 of the V1 deficit but only 0.07-0.19 of the
  V2/V3 deficit; median predicted/GT size ratio 0.92-1.05; frames with a GT side < 10 px: <= 1%.

## Not pursued in stage 3 (with reason)
- The 10-px minimum-size clamp (hift_tracker.py:68-69): V has <= 1% such frames, so it cannot be selected on V.
- Temporal / template update ("T1"; not R1, which is source-resolution fusion): the preregistered T-A threshold
  proposed by the reviewer (post-loss deficit share >= 0.40 overall) is not met (0.07-0.40 by subset).
- More SOT data (VisDrone-SOT train_part2): deferred until after this stage, needs a download/quota budget.

## Grid (configs/stage3_track_grid.json, 37 configurations)
"upstream" (config.yaml values) + WINDOW_INFLUENCE {0.15, 0.25, 0.35, 0.42} x LR {0.3, 0.5, 0.7} x
PENALTY_K {0.04, 0.08, 0.15} (minus the upstream point) + one score-matched point (0.23, 0.73, 0.08).
Implemented as whitelisted cfg.TRACK overrides (checkpoint_tracker.apply_track_overrides); the upstream
tracker code is unchanged. Grid entries without overrides are bit-identical to the plain path (smoke-tested).
Models: REF seeds 1-2 (TC2-P1-REF-001, TC2-P1-REF-S2-001) and B0 (sigmoid, NaN-safe), all on V1+V2+V3
(B0 never trained on UAVDT; the REF runs held the V3 videos out).

## Decision rule (REF seeds; selection on V only)
Select c* = the configuration with the highest seed-averaged M. Adopt c* ("REF/T") only if ALL hold:
1. seed-averaged dM(c*, upstream) >= +0.015 (stricter than 0.010: best of 37);
2. each seed: dM >= +0.010;
3. no subset (V1, V2-vehicle, V2-person, V3) drops by more than 0.015 seed-averaged;
4. the non-V1 part, mean(mean(V2-vehicle, V2-person), V3), does not decrease (seed-averaged);
5. leave-one-V1-sequence-out: removing any single V1 sequence keeps seed-averaged dM >= +0.008.
Reported but not a decision input: a two-fold split by video (select on one half, score on the other) as an
optimism estimate; B0 with the upstream config and with c* (attribution of the effect to weights vs
post-processing).

## Identity and sealed test
REF/T = B2 weights (REF) + tracker post-processing tuned on V; not a correction (not B1). If c* is adopted, the
sealed test is opened once for REF/T (both seeds) and B0-sigmoid/T; otherwise it stays closed for stage 3.
