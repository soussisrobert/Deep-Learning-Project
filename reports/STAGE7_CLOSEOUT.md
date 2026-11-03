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
