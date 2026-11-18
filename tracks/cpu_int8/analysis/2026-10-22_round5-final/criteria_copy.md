# Round 5: gated results (ROUND5_PLAN.md)
_Generated 2026-10-22 10:31_

## F/A. Failure and attack awareness from HiFT's own confidence

Failure = frame overlap < 0.1. Alarm = confidence below the 5th percentile of clean DEV frames.

- Alarm threshold (fixed from DEV clean): confidence < 0.9050
- **F0** DEV clean failure-warning AUROC 0.920 (2850 failed / 13324 good frames) → **PASS** (≥ 0.7)
- **A0** DEV standard PGD-10: 0.3% of 11303 attack-broken frames raise the alarm → **FAIL** (≥ 80%)
  - secondary signal map peak: DEV clean AUROC 0.921
  - secondary signal PSR: DEV clean AUROC 0.735

**Attack-alarm branch CUT at A0** (the alarm misses the standard attack on DEV).

- **Claim F** TEST clean failure-warning AUROC **0.968** [0.938, 0.989] (6075 failed / 31020 good frames) → **PASS** (≥ 0.75, interval above 0.5)
- TEST clean false-alarm rate at the DEV threshold: 7.1% (≤ 10% required for Claim A)

## K. Natural corruptions (severity 3 unless stated; synthetic night = darker + sensor noise)

| Corruption | fp32 GPU (Δ vs clean) | + JPEG-75 (Δ vs no defence) | int8bb-s1 CPU (Δ vs its clean) | int8 drop − fp32 drop | Failure-warning AUROC (fp32) |
|---|---|---|---|---|---|
| gauss_noise3 | 0.504 (-0.079 [-0.126, -0.036]) | 0.490 (-0.014 [-0.046, +0.007]) | 0.479 (-0.100 [-0.154, -0.049]) | -0.021 [-0.058, +0.008] | 0.955 |
| gauss_blur3 | 0.524 (-0.059 [-0.104, -0.020]) | 0.531 (+0.008 [+0.000, +0.018]) | 0.526 (-0.052 [-0.100, -0.011]) | +0.007 [-0.008, +0.021] | 0.978 |
| motion_blur3 | 0.552 (-0.031 [-0.073, +0.006]) | 0.532 (-0.019 [-0.039, -0.003]) | 0.542 (-0.036 [-0.078, -0.001]) | -0.006 [-0.025, +0.015] | 0.937 |
| contrast3 | 0.523 (-0.060 [-0.097, -0.026]) | 0.514 (-0.008 [-0.020, +0.001]) | 0.534 (-0.044 [-0.077, -0.013]) | +0.016 [-0.005, +0.040] | 0.892 |
| low_light3 | 0.555 (-0.027 [-0.069, +0.008]) | 0.541 (-0.014 [-0.035, -0.001]) | 0.545 (-0.033 [-0.070, -0.008]) | -0.006 [-0.032, +0.015] | 0.963 |
| low_light5 | 0.540 (-0.043 [-0.091, +0.002]) | 0.546 (+0.006 [-0.007, +0.025]) | 0.519 (-0.060 [-0.103, -0.022]) | -0.017 [-0.044, +0.005] | 0.946 |
| night3 | 0.504 (-0.078 [-0.126, -0.038]) | 0.509 (+0.005 [+0.000, +0.009]) | 0.503 (-0.076 [-0.124, -0.033]) | +0.003 [-0.031, +0.033] | 0.970 |

**Claim K** (int8 holds up as well as fp32 under bad conditions): mean extra drop -0.0035 over 7 corruptions → **PASS** (≥ -0.01)

## R4C. Round-4 confirmation: int8bb-s1 on all 123 videos, and losses by official UAV123 attribute

- 83 non-TEST videos (calibrated on the 40 TEST videos, exact recipe): 0.5936 vs fp32 CPU 0.5856: change +0.0080 [-0.0033, +0.0239]
  - sensitivity draw 1 (30 calibration videos): change +0.0078 [-0.0001, +0.0211]
  - sensitivity draw 2 (30 calibration videos): change +0.0022 [-0.0058, +0.0129]
- **Claim E-full** all 123 videos (TEST: mean of 3 draws; others: exact recipe): change **+0.0030** [-0.0054, +0.0140] → **PASS** (≥ −0.010); int8 0.5894 vs fp32 CPU 0.5865
- Largest losses: uav2 -0.155, car4_s -0.110, group1_2 -0.073, group3_2 -0.049, car1_2 -0.048

Official attribute labels (data/raw/UAV123/UAV.json). Descriptive, not a claim:

| Attribute | Videos | int8 change: with | without | difference [95%] |
|---|---|---|---|---|
| Aspect Ratio Change | 68 | +0.0030 | +0.0029 | +0.0002 [-0.0202, +0.0205] |
| Background Clutters | 21 | -0.0013 | +0.0038 | -0.0051 [-0.0197, +0.0062] |
| Camera Motion | 70 | +0.0034 | +0.0024 | +0.0011 [-0.0182, +0.0217] |
| Fast Motion | 28 | +0.0120 | +0.0003 | +0.0116 [-0.0147, +0.0506] |
| Full Occlusion | 33 | +0.0055 | +0.0021 | +0.0034 [-0.0212, +0.0383] |
| Illumination Variation | 31 | -0.0041 | +0.0054 | -0.0095 [-0.0260, +0.0034] |
| Low Resolution | 48 | +0.0020 | +0.0036 | -0.0017 [-0.0213, +0.0202] |
| Out-of-View | 30 | +0.0141 | -0.0006 | +0.0147 [-0.0082, +0.0497] |
| Partial Occlusion | 73 | +0.0012 | +0.0055 | -0.0043 [-0.0252, +0.0147] |
| Scale Variation | 109 | +0.0035 | -0.0010 | +0.0045 [-0.0054, +0.0173] |
| Similar Object | 39 | +0.0036 | +0.0027 | +0.0009 [-0.0199, +0.0254] |
| Viewpoint Change | 60 | -0.0034 | +0.0091 | -0.0125 [-0.0339, +0.0035] |

## Claims

- Claim F (HiFT's confidence warns of tracking failure): **SUPPORTED**
- Claim A (low-confidence alarm catches attacks, including an attacker who knows about it): **not met**
- Claim K (int8 keeps up with fp32 under natural corruptions): **SUPPORTED**
- Claim E-full (round-4 int8 recipe within −0.010 on all 123 videos): **SUPPORTED**

## NEXT (one GPU job at a time; a CPU queue may run beside it)

- nothing: round 5 is finished. Package the results.
