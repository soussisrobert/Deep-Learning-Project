# Stage 4 preregistration — long template-search gap in training pairs (S4-GAP), 2026-11-02

Written and committed before any stage-4 run exists. Direction from an independent Opus (xhigh) review of the
stage-2/3 results; its key measurement was re-checked here from the saved predictions.

## Why (measured on V, saved predictions; GT used only offline)
- V1 sequence-mean IoU by frame window [0,100) / [100,300) / [300,end) (9 sequences reach frame 300):
  B0-sigmoid 0.704 / 0.593 / 0.499; REF s1 0.742 / 0.619 / 0.485; REF s2 0.742 / 0.620 / 0.496;
  LONG s1 0.734 / 0.578 / 0.381. The fine-tune gain over B0 is near the template and gone after frame 300.
- Training pairs use a template-search gap of at most 100 frames (`TrackPairDataset(max_gap=100)`, never
  overridden by scripts/train_b2_domain.py through stage 3). Every V1 abrupt-loss onset is at frames 218-1430
  (reviewer measurement). Longer training on the same short-gap pairs (LONG) made late frames worse.
- Recipe knobs (stage 2) and post-processing (stage 3) never beat REF; this is a change of training-pair
  distribution, not of optimisation.
- Hypothesis (not established): short-gap-only pairs specialise the matcher to near-template appearance; pairs
  with long gaps teach appearance change over time. Confounded by sequence content (late frames are also harder).

## Arms (everything else = REF: corrected layout, ft-lr 1e-3 -> 1e-4, 10 x 1000 steps, batch 32, same sources,
## weights, 10% UAVDT holdout, search scale aug 0.18, upstream post-processing, sigmoid score + NaN-safe argmax)
- REF-C: REF re-run with identical arguments (control; GPU nondeterminism + code-path check).
- G300: `--max-gap 300`.
- GMIX: `--long-gap-probability 0.5` — half of the positive pairs draw the search frame uniformly from the whole
  valid track (src/hift_research/lasot/gap_policy.py); the other half keep max_gap 100.
Seeds: seed 1 = REF's (frozen config seed, data seed 20261027); seed 2 = REF-S2's (20261102 / 20261029).
6 runs, ~1.25 GPU-h each incl. in-job V (TC2 1x L40S, one at a time + one pending). Defaults reproduce the
stage-3 sampler and random stream exactly (tests/test_gap_policy.py).

## Decision rule (scripts/stage4_decision.py; selection on V only)
Gate: REF-C must reproduce the stored REF M within +-0.003 (seed-averaged); otherwise no adoption and the
discrepancy is reported as a reproducibility finding. An arm qualifies over REF-C only if ALL hold:
1. seed-averaged dM >= +0.015;  2. each seed dM >= +0.010;
3. no subset (V1, V2-vehicle, V2-person, V3) drops by more than 0.015 (seed-averaged);
4. non-V1 part mean(mean(V2veh, V2per), V3) changes by >= -0.005;
5. leave-one-V1-sequence-out: seed-averaged dM >= +0.008 for every left-out sequence;
6. paired episode bootstrap (stratified V1 / V2veh / V2per / V3, 2000 draws, seed 20261102): P10 of dM > 0.
If both qualify, adopt the larger dM. Mechanism check (reported; required to CLAIM the gap hypothesis, not to
adopt): V1 IoU on frames >= 300 rises vs REF-C in both seeds while [0,100) drops by < 0.01 (seed-averaged).

## Identity and sealed test
B2 retraining arm (training-pair distribution change), not a correction. If an arm is adopted the sealed test
(UAV123-domain 76 seqs, VisDrone-SOT test-dev 35) is opened once for that arm and REF-C (both seeds);
otherwise it stays closed for stage 4.

## Deferred (with reason, from the review)
- Half-pixel centre-offset question (+0.2 to +1.4 px median offset in every model): a B1 inference-geometry
  question, to be settled by a synthetic CPU round-trip test, never selected on V.
- Box-precision / IoU-aware score: needs a REF candidate-ceiling pass first; must not be combined with S4-GAP.
- VisDrone-SOT train_part2 (12.59 GB > ~10 GB TC2 headroom): needs a download approval + pruning plan; natural
  follow-up if S4-GAP helps. T1 template update: stage 5 if late-frame IoU rises but M misses the bar.
- R1 source resolution: existing data indicate a ceiling of ~+0.005 M; not justified. ResNet-18: rejected.
