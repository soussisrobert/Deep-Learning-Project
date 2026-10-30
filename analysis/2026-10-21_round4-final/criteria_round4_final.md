# Round 4: gated results (ROUND4_PLAN.md)
_Generated 2026-10-21 18:54_

## J. JPEG-85 (quality picked from round-3 DEV evidence; first TEST use)

- **J1 / H7 clean cost** (GPU, TEST): 0.5522, change -0.0303 [-0.0714, -0.0015] (+0/−3) → **FAIL** (≥ −0.010)
- Overhead on the A40 GPU (reported, not a round-4 claim): 1.53 ms = 14.4%

**JPEG-85 branch CUT at J1.**

## Q. int8 backbone recipes

### Q0 DEV pilot (TEST videos never used, not even for calibration)

| Recipe | Draw changes vs fp32 CPU (DEV) | Mean | Worst | Speed-up (4 thr) | Eligible |
|---|---|---|---|---|---|
| int8bb_c20x2 | -0.0015, +0.0215, -0.0004 | +0.0065 | -0.0015 | 2.27× | yes |
| int8bb_c40x4s | +0.0197, +0.0192, +0.0202 | +0.0197 | +0.0192 | 2.27× | yes |
| int8bb-s1_c20x2 | +0.0027, -0.0030, -0.0020 | -0.0008 | -0.0030 | 1.80× | yes |
| int8bb-s1_c40x4s | +0.0300, +0.0250, +0.0202 | +0.0251 | +0.0202 | 1.80× | yes |

Chosen on DEV: **int8bb-s1_c40x4s** (`--variant int8bb-s1 --calib-seqs 40 --calib-frames 4 --calib-spread`)

### Q1 TEST, 3 calibration draws (rule H9')

- draw 1: 0.5782, change vs fp32 CPU -0.0100 [-0.0237, -0.0011] (+0/−2)
- draw 2: 0.5792, change vs fp32 CPU -0.0091 [-0.0226, -0.0007] (+0/−2)
- draw 3: 0.5851, change vs fp32 CPU -0.0032 [-0.0068, -0.0006] (+0/−1)
- **mean -0.0074, SD 0.0037; speed-up 1.80× → PASS**

## C. int8 (chosen recipe) + JPEG-85 on CPU

- **CUT** (a JPEG or int8 gate failed)

## Claims

- Claim R-CPU (robust HiFT on CPU-class hardware): H6 + H7 + H8-CPU: **not met**
- Claim E (efficient HiFT): H9' + speed ≥ 1.5×: **IMPROVEMENT**
- Claim E+R (efficient and robust): all of the above + C1 + C2: **not met**

## NEXT (submit these, GPU/mix one at a time; CPU queues may run beside a GPU job)

- `plan/round4/queue/J1c_cpu.tsv`
