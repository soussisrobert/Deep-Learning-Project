# Round 3: results against the pre-registered rules (ROUND3_PLAN.md)

TEST = fixed 40 videos. Clean 0.582. Undefended PGD-10 0.095, PGD-50 0.090, worst per video 0.083.

## A. JPEG75 (round 2) under stronger adaptive attacks

| Attack on JPEG75 | Success AUC | Gain vs undefended (same strength) |
|---|---|---|
| BPDA, 10 steps (round 2) | 0.355 | +0.259 [+0.200, +0.320] (+34/−0) |
| surrogate gradient, 10 steps | 0.256 | +0.160 [+0.113, +0.211] (+27/−0) |
| surrogate gradient, 50 steps | 0.242 | +0.152 [+0.108, +0.198] (+28/−0) |
| BPDA, 50 steps | not run yet | – |
| **worst per video over the 3 attacks** | 0.234 | +0.151 [+0.108, +0.198] (+26/−0) vs worst undefended |

## B. Selected JPEG quality on TEST (rules H6, H7)

Chosen on DEV only: **jpeg75** (highest robustness among qualities with clean cost >= -0.010).

| Quality | DEV clean cost | DEV robust AUC | eligible |
|---|---|---|---|
| 75 | +0.0228 | 0.289 | yes |
| 85 | +0.0044 | 0.274 | yes |
| 90 | +0.0257 | 0.244 | yes |
| 95 | +0.0011 | 0.205 | yes |

Clean: 0.571 — change -0.012 [-0.029, +0.000] (+0/−2) → H7 (≥ −0.010): FAIL

| Attack on the selected JPEG | Success AUC | Gain vs undefended (same strength) |
|---|---|---|
| BPDA-10 | 0.355 | +0.259 [+0.200, +0.320] (+34/−0) |
| surrogate-10 | 0.256 | +0.160 [+0.113, +0.211] (+27/−0) |
| surrogate-50 | 0.242 | +0.152 [+0.108, +0.198] (+28/−0) |
| **worst per video (all 3)** | 0.234 | +0.151 [+0.108, +0.198] (+26/−0) vs worst undefended |

H6 (worst-case gain ≥ +0.10, interval above 0): PASS

GPU JPEG (tjpeg75): clean not run yet (–); surrogate-10 not run yet (–).

## C. Time overhead (rule H8: ≤ 3.85% of one tracking step on the same device)

| Defence | Device | Defence ms / frame | Tracking ms / frame (no defence) | Share | H8 |
|---|---|---|---|---|---|
| jpeg75 (round 2) | GPU (A40) | 1.46 | 10.64 | 13.7% | FAIL |
| jpeg75 | GPU (A40) | 1.46 | 10.64 | 13.7% | FAIL |
| tjpeg75 | GPU (A40), JPEG on GPU | not run yet | | | |
| jpeg75 + int8bb | CPU, 4 threads | 1.40 | 45.03 | 3.1% | PASS |

Benchmark cpu_t4_fp32: jpeg75 1.17 ms (1.3% of network); jpeg85 1.24 ms (1.4% of network); jpeg90 1.31 ms (1.4% of network); jpeg95 1.45 ms (1.6% of network); tjpeg75 3.80 ms (4.2% of network); tjpeg85 3.79 ms (4.2% of network); tjpeg90 3.79 ms (4.2% of network); tjpeg95 3.77 ms (4.2% of network); blur1 2.64 ms (2.9% of network); median3 26.16 ms (28.8% of network); bits5 0.22 ms (0.2% of network)

Benchmark cpu_t1_int8bb: jpeg75 1.22 ms (1.7% of network); jpeg85 1.29 ms (1.8% of network); jpeg90 1.35 ms (1.9% of network); jpeg95 1.47 ms (2.0% of network); tjpeg75 5.50 ms (7.6% of network); tjpeg85 5.48 ms (7.6% of network); tjpeg90 5.43 ms (7.5% of network); tjpeg95 5.46 ms (7.5% of network); blur1 3.25 ms (4.5% of network); median3 67.77 ms (93.4% of network); bits5 0.46 ms (0.6% of network)

Benchmark cpu_t4_int8bb: jpeg75 1.18 ms (2.9% of network); jpeg85 1.26 ms (3.1% of network); jpeg90 1.32 ms (3.2% of network); jpeg95 1.46 ms (3.6% of network); tjpeg75 3.89 ms (9.5% of network); tjpeg85 3.87 ms (9.5% of network); tjpeg90 3.94 ms (9.7% of network); tjpeg95 3.86 ms (9.5% of network); blur1 2.61 ms (6.4% of network); median3 26.20 ms (64.3% of network); bits5 0.22 ms (0.5% of network)

Benchmark cuda_t4_fp32: jpeg75 1.39 ms (25.5% of network); jpeg85 1.46 ms (26.7% of network); jpeg90 1.51 ms (27.7% of network); jpeg95 1.63 ms (29.9% of network); tjpeg75 1.33 ms (24.4% of network); tjpeg85 1.33 ms (24.4% of network); tjpeg90 1.33 ms (24.4% of network); tjpeg95 1.33 ms (24.4% of network); blur1 0.13 ms (2.5% of network); median3 1.21 ms (22.1% of network); bits5 0.07 ms (1.2% of network)

## D. int8 backbone: calibration spread (rule H9: mean change ≥ −0.010)

| Calibration | Success AUC (40) | Change vs fp32 CPU |
|---|---|---|
| round-2 set | 0.576 | -0.012 [-0.027, -0.001] (+0/−3) |
| set 1 | 0.567 | -0.021 [-0.051, -0.002] (+0/−3) |
| set 2 | 0.563 | -0.025 [-0.056, -0.004] (+0/−4) |
| set 3 | 0.583 | -0.005 [-0.017, +0.003] (+1/−2) |

Mean over 4 calibration sets -0.0161, SD 0.0089 → H9 FAIL

## E. int8 backbone + selected JPEG on CPU (rule H10)

- Clean: 0.571, change vs fp32 CPU -0.017 [-0.036, -0.002] (+0/−3) → FAIL (≥ −0.010)
- Speed: fp32 network 87.2 ms vs int8bb 41.2 ms + JPEG 1.40 ms → 2.05× → PASS (≥ 1.5×)

## F. Natural corruptions (descriptive; severity 3, ImageNet-C levels)

| Corruption | No defence (Δ vs clean) | With selected JPEG (Δ vs no defence) |
|---|---|---|
| gauss_noise3 | not run yet (–) | not run yet (–) |
| gauss_blur3 | not run yet (–) | not run yet (–) |
| motion_blur3 | not run yet (–) | not run yet (–) |
| low_light3 | not run yet (–) | not run yet (–) |
| contrast3 | not run yet (–) | not run yet (–) |

## Verdicts

- Claim R (robust HiFT, GPU deployment): H6 + H7 + H8 on GPU (OpenCV or GPU JPEG): **not met**
- Claim E+R (efficient and robust HiFT, CPU deployment): H6 + H9 + H10 (clean, speed, robust) + H8 on CPU: **not met**
