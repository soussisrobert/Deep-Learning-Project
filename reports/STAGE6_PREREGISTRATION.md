# Stage 6 preregistration — V1B: more validation power (no training), 2026-11-03

Written and committed before the job runs. Owner choice 2026-11-03: "Bigger validation". No arm is adopted in this
stage, and the sealed test stays closed.

## Why (stage-6 review, offline analysis of saved V predictions; reviewer's numbers, re-checked where noted)
- Stages 3-5 rejected every arm, but V may not be able to resolve the effects we look for. The review measured V
  sampling SE of dM at 0.004-0.014 (episode and video-cluster bootstraps), while the seed SE is only about 0.0015.
  The adoption bar was +0.010, so more seeds do not help; more validation sequences do.
- V1 has 11 sequences from 9 videos. Three of them account for 91% of the V1 variance across runs, and the stage-4
  V1 losses (G300 -0.029, GMIX -0.042) come almost entirely from one sequence per arm.
- Measured here (reports/SOURCE_VIDEO_AUDIT.md): SOT train_part2 has 35 sequences from 31 source videos that share no
  source video with MOT-train, SOT part1, SOT-val or SOT test-dev. B0, REF and the stage 2-4 arms never trained on
  part2, so these 35 can serve as extra long-term SOT validation (V1B), about 3x V1.

## Job (inference only; hift_s6_v1b, scripts/validate_domain.py --subsets V1 V1B)
- V1B = configs/v1b_sequences.txt (35 names; checked against the audit lists by tests/test_v1b.py), full length,
  initialised from GT at the first usable frame like V1.
- Models: B0 (official checkpoint, sigmoid, NaN-safe), REF seeds 1-2 (TC2-P1-REF-001 / -S2-001), stage-4 arms G300
  and GMIX, seeds 1-2. Their final-model.pt files are re-uploaded with sha256 verification to ~/hift/checkpoints/,
  because their run dirs were pruned. SOT2 is excluded, since it trained on part2.
- Gate: V1 is repeated in the same job. Every model's per-episode V1 AUC must equal its stored value (|diff| < 1e-9:
  B0 from the stage-2 record, the others from their run records). If the gate fails, V1B results are unreliable
  and only the failure is reported.

## Analyses (scripts/stage6_v1b.py; source-video cluster bootstrap, 2000 draws, seed 20261104)
1. Noise: REF V1B per seed and the seed difference; the bootstrap SE of arm - REF on V1B, SOT-long (V1 + V1B,
   46 sequences) and M'.
2. Replication of the stage-4 rejections. For G300 and GMIX, the seed-averaged dV1B vs REF with P10/P90. An arm
   with dV1B >= +0.015 and P10 > 0 is reported as a **V/V1B disagreement**. Its rejection then needs revisiting,
   but nothing is adopted here; any sealed-test question goes to the owner. P90 < 0 means V1B confirms the arm is
   worse.
3. Fine-tuning check: B0 -> REF on V1 (+0.013 known) vs V1B, against the sealed test-dev gain on the 34 disjoint
   sequences (+0.075). This is descriptive: does V1B track the sealed direction better than V1?

## Fixed now for stage 7 and later (before seeing V1B)
- Primary metric M' = mean(SOT-long, mean(V2-vehicle, V2-person), V3), where SOT-long is the sequence-macro AUC over
  V1 + V1B (46 sequences). Old M is reported alongside.
- Bootstraps and leave-one-out resample source videos, not episodes.
- Stage-7 thresholds are set in its own preregistration from the SEs measured here. This stage does not test a
  new training idea.

## Budget and disk
- One validation job: 7 models x (V1 + V1B, about 40k frames) ≈ 35-45 min on one GPU.
- Uploads 4 x 44 MB. TC2 home is at 98.5% of 100 GiB. After retrieval and analysis, part2 (12.6 GiB) is deleted
  (owner-approved), along with the 4 re-uploaded checkpoints.

## Limitations
- V1B shares source video uav0000182 with V2 (MOT-val), so they are not fully independent. Part2 sequences are
  VisDrone-SOT "train" annotations; we have not checked that their quality matches SOT-val's.
- Only exact source-ID overlap is excluded. Visually similar scenes from different flights remain.
