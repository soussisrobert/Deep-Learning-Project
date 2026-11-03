# Stage 5 preregistration — more SOT training data (VisDrone-SOT train_part2), 2026-11-03

Written and committed before the data job or any stage-5 run. Owner choice 2026-11-03: "More SOT data"; TC2 space
plan approved ("Limit is 100 GiB, go").

## Why (measured on V)
- Stages 2-4 changed optimisation (LR, frozen backbone, longer training), pair sampling (distractor negatives,
  long template-search gaps) and post-processing; none beat REF. The stage-4 control showed REF retraining is
  bitwise reproducible, so these were real negatives, not noise.
- V1 (VisDrone-SOT-val) is the weakest subset: REF seed-avg V1 0.552 vs V2-vehicle 0.708, V2-person 0.612, V3 0.732.
- REF trains on VisDrone-SOT train_part1 only (43 sequences). train_part2 is 43 more sequences (40,622 frames,
  12.62 GiB). Hypothesis (not established): SOT appearance diversity, not the recipe, limits V1.

## Arm (everything else = REF: corrected layout, ft-lr 1e-3 -> 1e-4, 10 x 1000 steps, batch 32, source weights
## SOT 0.3 / MOT 0.25 / UAVDT 0.2, 10% UAVDT holdout, max gap 100, upstream post-processing, sigmoid + NaN-safe)
- SOT2: `--train-parts visdrone-sot-1 visdrone-sot-2 visdrone-mot uavdt`. Source weights are unchanged, so the SOT
  share of pairs stays 40%; each SOT track is drawn about half as often. This isolates diversity, not SOT volume.
- Seeds: REF's two pairs (config/data seed 20261023/20261027 and 20261102/20261029). Runs TC2-S5-SOT2-001 and
  TC2-S5-SOT2-S2-001.
- Control: REF (TC2-P1-REF-001, TC2-P1-REF-S2-001) as stored. No rerun: stage 4 reproduced REF's final-model.pt
  byte for byte on both seeds.
- Expected in-job check: train_tracks = 7478 (REF) + the usable part2 tracks (at most 43); manifest train_parts
  must list visdrone-sot-2.

## Data identity and leakage
- Source: Hugging Face huseyincavus/visdrone2019-sot, visdrone2019-sot.zip (Hub LFS sha256 1dbbcdbe...c182, the
  same archive as part1), members under VisDrone2019-SOT-train_part2/; HTTP range reads with per-member CRC-32
  check (scripts/acquire_hf_zip_members.py); immutable ledger ~/hift/ledgers/SOT2.json.
- Measured 2026-11-03 from the archive index: 43 part2 sequences, 0 name overlap with part1, SOT-val (11) or SOT
  test-dev (35).
- In the data job: byte-identical frame check of part2 against V1 (SOT-val) and V2 (MOT-val)
  (scripts/frame_overlap.py). Before any sealed test: the same check locally against SOT test-dev, using the part2
  hash list. Any duplicate means no adoption and an investigation. Limitation: only exact copies are detected.

## Decision rule (scripts/stage5_decision.py; selection on V only)
SOT2 is adopted over REF only if there are no leaked frames and ALL of these hold:
1. seed-averaged dM >= +0.010;  2. each seed dM >= +0.005;
3. no subset (V1, V2-vehicle, V2-person, V3) drops by more than 0.015 (seed-averaged);
4. non-V1 part mean(mean(V2veh, V2per), V3) changes by >= -0.005;
5. leave-one-V1-sequence-out: seed-averaged dM >= +0.005 for every left-out sequence;
6. paired episode bootstrap (stratified V1 / V2veh / V2per / V3, 2000 draws, seed 20261103): P10 of dM > 0.

Threshold rationale (fixed before any run): across the 7 earlier two-seed arms the per-seed dM differed by at
most 0.007 (median 0.003), so the 2-seed mean has an SE of about 0.003; +0.010 is the stage-2 bar. Stage 4 used
+0.015 / +0.010 for a mechanism-motivated sampler; a data addition that mainly targets V1 (one third of M) gets
the stage-2 bar plus the stage-4 robustness checks (3-6).

Descriptive only: V1 IoU change by frame window. REF's per-frame predictions come from REF-C, whose weights are
identical.

## Identity, sealed test, budget
- B2 retraining arm (training-data change), not a correction.
- If adopted: run the local test-dev leakage check, then open the sealed test (UAV123-domain 76 sequences,
  VisDrone-SOT test-dev 35) once for SOT2 seeds 1-2. REF's stage-2 sealed results stand, since its weights are
  identical. If rejected, the sealed test stays closed.
- TC2: 1 data job (about 1 h; it holds one GPU slot because the partition requires one) + 2 training runs (about
  1.3 h each incl. in-job V). Space: 30 old HiFT runs + 5 code copies pruned (89.2 -> 84.4 GiB); about 2.9 GiB is
  left after part2 under the 100 GiB limit.

## Deferred
- T1 template update: its trigger (late-frame V1 IoU rises under S4-GAP) was not met.
- Half-pixel centre-offset question (B1 geometry, CPU synthetic test); IoU-aware score after a candidate-ceiling
  pass; R1 source resolution (ceiling about +0.005 M). Unchanged from stage 4.
