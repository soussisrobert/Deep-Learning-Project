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

## Outcome (2026-11-03; job 50675, 2854 s, exit 0)
- Run: bundle 8e3ca6c (payload sha256 b8bdcb07...), job hift_s6_v1b 50675 on TC2, 7 models x (V1 + V1B). Results were
  retrieved as one sha256-verified tar into artifacts/tc2-validation/V1B-<label>/. The analysis is in
  artifacts/stage6/stage6_v1b.json. validation.json sha256 prefixes: B0 f5846ab4, REF 8c4b74b4 / 237c8e85,
  G300 06c6cb9c / d33e1511, GMIX c0af4768 / 5dd28ce2.
- **Gate: PASS.** For all 7 models, every one of the 11 per-episode V1 AUCs equals the stored value (|diff| < 1e-9).
  There is one episode-listing hash, and V1B has 35 episodes from 31 source videos.

| model | V1 | V1B | SOT-long | V2veh | V2per | V3 | M | M' |
|---|---|---|---|---|---|---|---|---|
| B0 (official) | 0.5387 | 0.5124 | 0.5187 | 0.6984 | 0.5745 | - | - | - |
| REF s1 / s2 | 0.5495 / 0.5549 | 0.6041 / 0.5928 | 0.5910 / 0.5838 | 0.7091 / 0.7067 | 0.6117 / 0.6117 | 0.7334 / 0.7313 | 0.6627 / 0.6634 | 0.6616 / 0.6581 |
| G300 s1 / s2 | 0.5227 / 0.5237 | 0.5868 / 0.5992 | 0.5715 / 0.5811 | 0.7147 / 0.7178 | 0.6116 / 0.6118 | 0.7179 / 0.7380 | - | 0.6508 / 0.6613 |
| GMIX s1 / s2 | 0.5105 / 0.5101 | 0.5940 / 0.6057 | 0.5740 / 0.5829 | 0.7195 / 0.7194 | 0.6053 / 0.6076 | 0.7279 / 0.7240 | - | 0.6548 / 0.6568 |

(M is shown for REF only, as a cross-check against the stage-2 record. B0's stage-2 record has no V3, so B0 has no
M or M'.)

### Noise (measured)
- Seed-to-seed noise was estimated from 3 seed pairs (REF, G300, GMIX), as the pooled per-seed SD: V1 0.0023,
  V1B 0.0083, SOT-long 0.0061, M 0.0031, M' 0.0046. V1B, which consists of long sequences, is the most
  seed-sensitive subset. The review's estimate of about 0.0015 seed SE understates seed noise on M'. These are only
  3 pairs, so the estimates are rough.
- Video-cluster bootstrap SE of arm - REF, using 2-seed means: V1B 0.005-0.013, SOT-long 0.007-0.014,
  M' 0.003-0.006.

### Replication of the stage-4 rejections (seed-averaged arm - REF)
- G300: dV1 -0.0290, dV1B -0.0055 (P10/P90 -0.0121/+0.0012; per-seed -0.0173/+0.0063), dSOT-long -0.0111,
  dM' -0.0038 (P10/P90 -0.0075/+0.0000). There is no V/V1B disagreement. V1B does not confirm the arm is worse,
  because P90 >= 0. **The rejection stands.**
- GMIX: dV1 -0.0419, dV1B +0.0014 (P10/P90 -0.0157/+0.0176; per-seed -0.0101/+0.0129), dSOT-long -0.0090,
  dM' -0.0041 (P10/P90 -0.0113/+0.0031). There is no disagreement. **The rejection stands.** The large V1 loss does
  not replicate on V1B, so V1 exaggerated it, but there is no gain either.

### Fine-tuning check (descriptive)
- B0 -> REF: V1 +0.0135, V1B +0.0861 (B0 - REF bootstrap P10/P90 -0.1194/-0.0517). The stage-2 sealed test-dev
  gain on the disjoint 34 sequences was +0.0749. V1B tracks the sealed gain, while V1 understated it about 6x. This
  is a single comparison, so it is evidence that V1B is the more representative SOT validation set, not proof.

### Disk
- The gate passed, so SOT train_part2 and ~/hift/checkpoints were deleted (owner-approved). TC2 home is now
  86988 MiB (about 85 GiB of 100 GiB).

### Inputs for the stage-7 preregistration (measured here, not decisions)
- Compare two 2-seed means on M': seed noise contributes about 0.0046 x sqrt(2/2) = 0.0046 and the video bootstrap
  about 0.003-0.006, so the total SE is roughly 0.006-0.008. A +0.010 bar is only about 1.3-1.7 SE. Clearing it
  reliably needs at least 3-4 seeds per arm and a paired REF re-run, or a larger effect.

### Limitations
- Seed noise comes from 3 pairs only. V1B annotation quality has not been checked against SOT-val. V1B shares
  source video uav0000182 with V2.
