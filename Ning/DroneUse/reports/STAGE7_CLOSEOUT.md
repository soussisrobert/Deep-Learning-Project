# Stage 7 — close-out: freeze REF, sealed-test confidence intervals, 2026-11-04

Written and committed before the analysis runs. Owner choice 2026-11-04: "1 Freeze REF (Recommended)", one of the
three options in the stage-7 review. This stage runs no training and no inference. The sealed test is not reopened:
the analysis only re-reads the stage-2 sealed predictions already stored on disk, and nothing is selected on them.

## Decision
REF is the frozen final model of the domain-tracking line:
- Runs: TC2-P1-REF-001 (seed 20261023) and TC2-P1-REF-S2-001 (seed 20261102).
- Recipe: B2-L1 (weight 1.0) head with the AlexNet backbone, aerial vehicles + pedestrians, 10000 updates, last
  checkpoint (git 38de4b2d).
- final-model.pt sha256: 88fef3aa4be46eab... (s1) and c475d0372d452c35... (s2).

No further arms are trained or screened. Options 2 (re-detection diagnostic) and 3 (more REF seeds + weight soup)
from the stage-7 review are not run.

## Inputs (stored, read-only)
- artifacts/b2-eval/STAGE2-SEALED-{VISDRONE,UAV123}-{B0,TC2-P1-REF-001,TC2-P1-REF-S2-001}-sigmoid: the evaluation
  JSON, prediction CSVs and inference manifest for each. All six were produced on 2026-10-30/31 with git c7a27cc,
  sigmoid scores and NaN-safe selection. The evaluator sources have not changed since then.
- Dataset manifests: artifacts/domain-visdrone/dataset_manifest.json (VisDrone2019-SOT-test-dev, 35 sequences) and
  artifacts/domain-uav123/ACQUISITION-002/dataset_manifest.json (UAV123@10fps domain subset, 76 sequences, partial).

## Gate (all must pass before any interval is reported)
- G1: each prediction CSV's sha256 equals the hash recorded in its evaluation JSON.
- G2: each dataset manifest's sha256 equals the recorded one.
- G3: each inference manifest's checkpoint sha256 equals the local file. For REF that is the run's final-model.pt;
  for B0 it is the official artifacts/checkpoints/general_model.pth (8c697d2b...).
- G4: re-scoring every CSV against the manifest ground truth with the unchanged evaluator (`evaluate_sequence`,
  count_as_failure) reproduces every stored per-sequence success AUC and precision@20px, and the sequence-macro
  aggregate, within 1e-12.

If any gate fails, only the failure is reported.

## Analysis (scripts/stage7_closeout.py; output artifacts/stage7/closeout.json)
- Test sets:
  - VisDrone test-dev disjoint-34 is primary. It excludes uav0000079_02568_s, whose source video is in REF's
    training data (reports/SOURCE_VIDEO_AUDIT.md). All-35 is reported alongside.
  - UAV123@10fps domain subset (76 sequences).
- Metrics: sequence-macro success AUC (primary) and precision@20px (secondary).
- Contrasts:
  - REF − B0, where REF is the per-sequence mean of the two seeds (primary);
  - each seed − B0;
  - seed 1 − seed 2.
- Intervals: a paired source-video cluster bootstrap. Each draw resamples clusters with replacement and applies the
  same sequences to both models. 10000 draws, `numpy.random.default_rng(20261105)`. Reported: the 95% percentile
  interval (P2.5/P97.5), P10/P90 (as in earlier stages), the bootstrap SE, and the leave-one-cluster-out minimum and
  maximum of the delta.
- Clusters:
  - VisDrone: the source flight video `uavNNNNNNN`, i.e. the first 10 characters (22 videos in disjoint-34, 23 in
    all-35).
  - UAV123: the sequence name with a trailing `_<k>` removed, so car1_1..car1_3 become car1 (48 clusters).

## Interpretation rules (fixed now)
- "REF improves over B0 on a set": the 95% lower bound of the REF − B0 delta is > 0, and both seed deltas are > 0.
- "REF is worse than B0 on a set": the 95% upper bound is < 0.
- Otherwise: "no difference detectable at this sample size", reported with the interval width. It is never stated
  as "no effect".
- Claims cover these checkpoints, predictions and subsets only. This is not a checkpoint-reproduction claim, and
  not comparable with published UAV123 or VisDrone leaderboards: the UAV123 set is a partial domain subset, and
  official membership is not verified for either set.

## Final report contents
This file, appended after the run:
- the gate result and sealed-test table with intervals;
- the validation trajectory of stages 2-6 (from existing records);
- the stage-7 review's failure-mode findings, labelled as review-measured. The review's scratch scripts stayed on
  the Windows machine's temp folder and are not in the repo; they are not re-run here;
- limitations and open questions.

## Known limitations (before the run)
- Only 22-23 independent VisDrone videos and 48 UAV123 clusters, so intervals will be wide. Only 2 seeds.
- UAV123 clustering is by name only. Different numbered sequences may still come from similar scenes.
- The official B0 checkpoint's pretraining membership is unknown, and may include UAV123.
- The sealed sets were opened once at stage 2. This re-analysis cannot change, and does not revisit, the stage-2
  selection of REF.

## Outcome (2026-11-04; CPU only, no inference)
- Run: `python scripts/stage7_closeout.py` at commit dde0575 on Arch Linux (Python 3.13.18, NumPy 2.4.4), in the
  Linux clone at ~/Code/HiFT. Output: artifacts/stage7/closeout.json (sha256 933d3e3ee15d0d2a...).
- **Gate: PASS** for all 6 stored runs:
  - G1: all prediction-CSV and inference-manifest hashes match;
  - G2: both dataset manifests match;
  - G3: checkpoints match (B0 8c697d2b..., REF 88fef3aa... / c475d037...);
  - G4: every per-sequence success AUC and precision@20px, and every aggregate, re-scored within 1e-12.

### Sealed test: REF vs B0 (paired source-video cluster bootstrap, 10000 draws)
Success AUC, sequence-macro. REF is the per-sequence mean of seeds 1 and 2.

| set | seqs / clusters | B0 | REF s1 / s2 | REF − B0 | 95% interval | LOCO min / max | verdict |
|---|---|---|---|---|---|---|---|
| VisDrone test-dev disjoint-34 (primary) | 34 / 22 | 0.5267 | 0.6032 / 0.6000 | **+0.0749** | [+0.024, +0.124] | +0.053 / +0.081 | REF improves over B0 |
| VisDrone test-dev all-35 | 35 / 23 | 0.5321 | 0.6067 / 0.6035 | +0.0730 | [+0.024, +0.122] | +0.051 / +0.078 | REF improves over B0 |
| UAV123@10fps domain subset | 76 / 48 | 0.5935 | 0.5826 / 0.5945 | −0.0050 | [−0.034, +0.027] | −0.013 / +0.001 | no difference detectable |

- Per-seed deltas, disjoint-34: s1 +0.0765 [+0.025, +0.127], s2 +0.0733 [+0.023, +0.121]. UAV123: s1 −0.0109
  [−0.039, +0.021], s2 +0.0009 [−0.030, +0.036].
- Precision@20px (secondary), REF − B0:
  - disjoint-34: +0.093 [+0.007, +0.174], read as improves;
  - UAV123: +0.026 [−0.019, +0.074], no difference detectable.
- Seed spread:
  - VisDrone: s1 − s2 = +0.0032, with a 95% interval of [−0.000, +0.009].
  - UAV123: s1 − s2 = −0.0119 [−0.033, +0.001]. This is about twice the REF − B0 point estimate, so the UAV123
    result is dominated by seed noise.
- Reading:
  - REF's VisDrone gain over the official checkpoint is well clear of zero, under both source-video resampling
    and leave-one-video-out. The interval is wide: with 22 videos, the gain lies anywhere from about +0.02 to
    +0.12.
  - On the UAV123 domain subset the data cannot tell REF from B0. With a 95% interval width of 0.061, a change
    of about ±0.03 is still possible. This is not evidence of "no effect".

### Validation trajectory (stages 2-6, from the existing records; nothing re-run)
- Stage 2:
  - P1 fine-tune strength: LR3E4, FROZEN and LONG were all rejected against REF. REF has M = 0.648 / 0.649.
  - P2 same-video distractor negatives: D10 and D20 were rejected.
- Stage 3: tracker post-processing grid on frozen weights. The upstream config was the best of 37 settings, so
  REF/T was rejected.
- Stage 4: long template-search gap sampler. G300 and GMIX were rejected. REF-C reproduces REF bit for bit, so
  training on TC2 is deterministic for a fixed seed pair.
- Stage 5: adding VisDrone-SOT train_part2 (SOT2) gave dM −0.0001, so it was rejected.
- Stage 6: the V1B extra validation subset passed its gate.
  - The stage-4 rejections stand on V1B and M'.
  - V1B tracks the sealed B0 → REF gain (+0.086 vs +0.075 sealed); V1 does not (+0.013).
  - M' seed SD is 0.0046.

In total, eight training arms (LR3E4, FROZEN, LONG, D10, D20, G300, GMIX, SOT2) and a 37-point post-processing
grid were tested against REF. None met its preregistered adoption rule.

### Failure modes (stage-7 review, review-measured; not re-run here)
These figures come from a read-only review on 2026-11-03, using REF-C per-frame outputs on V1/V2/V3. Its scratch
scripts stayed in the Windows temp folder and are not in this repo. Treat them as diagnostic leads, not as
reproduced results.
- About 62% of REF's M' deficit is box imprecision on frames where the target is tracked (IoU ≥ 0.5), 22% is lost
  frames, and 16% is partial overlap.
  - The imprecision is random per frame, not a bias. The best output-only rescale gains +0.004, and causal centre
    smoothing loses up to 0.013.
  - Oracle centre and oracle size on non-lost frames would add +0.06 to +0.12 and +0.04 to +0.05 respectively.
    This points to localizer capacity, not post-processing.
- Losses rarely recover (V2-vehicle 0/6; V1 1/7).
  - On V2/V3 lost frames, the box sits on another annotated object 55-93% of the time, sometimes at score 0.95+.
  - The score detects loss only moderately (AUROC 0.75-0.91).
- Five of the 46 SOT-long sequences score below 0.3 for every trained model, and carry 23% of REF's SOT-long
  deficit. Seed noise is mostly loss events that flip between seeds.

### What this establishes, and what it does not
- Established (measured): within this project's evaluator, REF fine-tuned from the official HiFT checkpoint
  improves success AUC on VisDrone-SOT test-dev over the checkpoint itself. The gain is +0.075 on the 34 sequences
  whose source videos are not in training, 95% cluster interval [+0.024, +0.124], with both seeds positive.
- Not established:
  - Any UAV123-domain gain.
  - Any improvement over published trackers or leaderboard numbers: there was no official-protocol run, and
    membership is unverified.
  - Reproduction of the official checkpoint's published results.
  - The source-resolution hypothesis (R1). R1 was never run. Per AGENTS.md it needs a preregistered R1-A diagnostic
    first, and none exists.

### Limitations
- Two seeds; 22 independent VisDrone videos; UAV123 clusters by name only.
- V1B annotation quality was never checked against SOT-val.
- The official checkpoint's pretraining data is unknown and may include UAV123 or VisDrone scenes.
- The sealed sets were opened once at stage 2, for B0 and REF only. Arms rejected on validation were never
  sealed-tested, which is by design.

### Open questions (not authorized here)
- Whether box precision (localizer capacity) can be raised without a large backbone download. The past ResNet-18
  attempts did not help.
- Whether the five hard-floor sequences are annotation problems. Answering this needs the V1B annotation files,
  which were deleted from TC2 together with part2.
