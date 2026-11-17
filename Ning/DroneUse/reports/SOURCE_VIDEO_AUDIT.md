# VisDrone source-video audit, 2026-11-03

Required by docs/datasets.md rule 3 ("record ... source-video mappings") but not done before this date. Prompted by the
stage-6 review. Method: a VisDrone sequence name `uavNNNNNNN_FFFFF_{s,v}` encodes the source flight video
(`uavNNNNNNN`) and its start frame. Two splits overlap when they share a source-video ID; the segments may differ,
but the scene and possibly the objects are shared. Name lists: reports/source_video_audit_names.txt (TC2
`ls` of ~/hift/data/visdrone, train/val splits; local SOT test-dev). Byte-level frame hashing (stage 5) cannot
detect this kind of overlap.

| split | sequences | source videos |
|---|---|---|
| MOT-train (REF training) | 56 | 48 |
| SOT train_part1 (REF training) | 43 | 27 |
| SOT train_part2 (stage-5 SOT2 only) | 43 | 39 |
| SOT-val (V1) | 11 | 9 |
| MOT-val (V2) | 7 | 7 |
| SOT test-dev (sealed) | 35 | 23 |

## Overlaps (measured)
- **Sealed test-dev vs REF training: 1 video.** uav0000079 is in MOT-train and in test-dev sequence uav0000079_02568_s
  (1 of 35). Test-dev shares no video with SOT part1 or part2.
- V1 and V2 vs REF training (MOT-train, part1): none.
- V1 and V2 share uav0000086. Both are validation, so this is not leakage, but V1 and V2 are not fully independent.
- Part2 vs V2: uav0000182, in the rejected SOT2 arm only. Part2 vs MOT-train: 8 videos (8 sequences).
- Not covered: UAVDT (V3 split by video in-job), UAV123-domain vs training (different datasets, not checked by
  name), and pretraining data of the official checkpoint (unknown membership, docs/datasets.md).

## Stage-2 sealed VisDrone test-dev, re-scored on the 34 disjoint sequences
This re-scores the existing stage-2 predictions (artifacts/b2-eval/STAGE2-SEALED-VISDRONE-*); it is not a
re-opening and runs no new inference. Metric: sequence-macro success AUC.

| model | all 35 | disjoint 34 | uav0000079_02568_s |
|---|---|---|---|
| B0 (sigmoid, NaN-safe) | 0.5321 | 0.5267 | 0.7153 |
| REF seed 1 | 0.6067 (+0.0746) | 0.6032 (+0.0765) | 0.7255 |
| REF seed 2 | 0.6035 (+0.0714) | 0.6000 (+0.0733) | 0.7203 |

The REF gain over B0 holds on the disjoint subset (seed average +0.075 vs +0.073), so the overlapping sequence does
not drive it. From now on the disjoint-34 subset is primary for test-dev claims; all-35 is reported alongside.
