# Stage 2 preregistration — drone tracking (vehicles + people), 2026-10-30

Written and committed before any stage-2 run or diagnostic result exists.

## Status of stage-1 evidence (exploratory)
Every stage-1 choice (score rule, layout, fine-tune vs scratch, scale augmentation) was compared on the two test
sets (UAV123 domain subset, VisDrone-SOT test-dev). Stage-1 rankings are therefore exploratory, not confirmed.
Reviewer findings that change the reading of stage 1 (verified):
- The B0 test row used raw score + upstream argmax; all fine-tuned rows used sigmoid + NaN-safe selection.
- All B0-FT "seeds" share identical initial weights (`initial_parameters_sha256` 32f52738…); seeds only change
  dropout/data order, so they are not independent evidence.
- The corrected-vs-original VisDrone gain (+0.025 AUC) is concentrated in one sequence (uav0000367_02761_s);
  the layout's accuracy effect is not established. The corrected layout stays because it fixes a bug.
- Stage-1 B0-FT manifests record `backbone_unfreeze_epoch: 10`, but backbone layers 3-5 trained from step 0.

## Validation set V (model selection only)
Implemented in src/hift_research/evaluation/validation.py and scripts/validate_domain.py (seed 20261030):
- V1: VisDrone2019-SOT-val, whole sequences.
- V2: VisDrone2019-MOT-val single-object episodes; 40 vehicle (car/van/truck/bus) + 40 person
  (pedestrian/people), at most 7 per video, start at the first valid box >= 10 px, at most 200 frames.
- V3: episodes from 10% of UAVDT videos (deterministic holdout), excluded from training in stage-2 runs;
  40 vehicle episodes, at most 8 per video.
Scored on valid-GT frames only, success AUC (21 thresholds), sequence-macro.
Primary metric M = mean(V1 AUC, mean(V2 vehicle, V2 person), V3 AUC).
Neither val split nor the V3 videos are trained on in stage 2.

## Sealed test
UAV123 domain subset + VisDrone-SOT test-dev are opened once at the end of stage 2, for: B0 (raw/upstream and
sigmoid/NaN-safe), the P1 reference arm, and the selected arm. Nothing is selected on them.

## Diagnostics before training (D1, D2; validation only, V1+V2)
B0 raw/upstream, B0 sigmoid/NaN-safe, and the six stage-1 B0-FT final models (orig/corr x 3 seeds).
Purpose: check whether V reproduces the stage-1 test ordering, and measure V noise.

## P1 — fine-tune strength / forgetting (corrected layout, UAVDT holdout 0.1, 2 seeds each)
Hypothesis: a gentler fine-tune keeps cross-domain generality while keeping the VisDrone gain.
- REF: --ft-lr 1e-3 1e-4, 10 x 1000 steps (stage-1 recipe, re-run under the holdout)
- LR3E4: --ft-lr 3e-4 3e-5, 10 x 1000 steps
- FROZEN: --ft-lr 1e-3 1e-4, backbone frozen (BN eval), 10 x 1000 steps
- LONG: --ft-lr 1e-3 1e-5, 10 x 3000 steps
Decision rule: adopt an arm over REF only if the seed-averaged dM >= +0.010, BOTH seeds have dM > 0, and no V
subset (V1, V2-vehicle, V2-person, V3) drops by more than 0.015 seed-averaged. If several qualify, take the
largest seed-averaged dM. Otherwise REF stays the stage-2 model.

## Deferred
P2 (same-video distractor negatives), P3 (category rebalancing), P4 (ResNet-18 with an annealed schedule)
depend on the P1 outcome. R1/template-update work stays deferred per AGENTS.md.
