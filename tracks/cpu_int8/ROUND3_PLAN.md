# Round 3 plan: turn the two near-misses into a conclusive answer

Owner: Manzi (adversarial + edge tracks). **Status: prepared, not run.**

The rules below are fixed before any round-3 result exists. `scripts/evaluate_round3.py` applies them mechanically, and the thresholds are the round-2 ones, unchanged.

- **Why these tests:** `analysis/2026-10-19_round2-final/REPORT.md`.
- **Queues:** `plan/round3/SCHEDULE.md`.
- **How to run them:** `AGENT_RUNBOOK_ROUND3.md`.

## 1. Questions

Round 2 left two candidates:

| Candidate | Round-2 result | What stopped it |
|---|---|---|
| **JPEG-75 input filter** | Recovered 53% of the attack loss | Clean cost −0.0116 (limit −0.010). Overhead 13.7% on the A40 (limit 3.85%). Adaptive attack only 10-step BPDA |
| **int8 backbone** | 2.1× faster; unchanged on all 123 videos | −0.0125 on the 40-video test set |

Round 3 asks:

1. **Robustness:** does JPEG still help against much stronger adaptive attackers?
2. **Clean cost:** is there a JPEG quality that keeps the clean cost within 0.010?
3. **Overhead:** can the filter's time fit HiFT's real-time margin, either on the GPU or next to the int8 model on a CPU?
4. **Calibration:** is the int8 miss real, or just calibration noise?
5. **Combination:** do int8 and JPEG work together?

## 2. Design

**Video sets:**
- **TEST:** the fixed 40 (`attack-subset-40.json`), as in round 2.
- **DEV:** 20 different UAV123 videos (`plan/round3/dev-20.json`). They are not in TEST and not in the round-2 int8 calibration set. DEV is used **only** to choose the JPEG quality, so nothing is tuned on TEST.

**Two stages:**
- **Stage 1:**
  - JPEG quality sweep (75, 85, 90, 95) on DEV, under the selection attack: PGD-10, ε = 4, adaptive, surrogate gradient.
  - Stronger attacks against the round-2 JPEG-75 on TEST.
  - Defence timing benchmarks.
- **Selection:** `scripts/r3_select.py` picks the quality from DEV only, using this rule:
  1. A quality is eligible if its DEV clean cost is ≥ −0.010.
  2. Among eligible qualities, pick the highest DEV robustness (ties: the higher quality).
  3. If none is eligible, pick the smallest clean cost.
- **Stage 2:** the selected quality ("sel") on TEST:
  - clean, plus three adaptive attacks and seeds;
  - the same JPEG done on the GPU;
  - int8 + JPEG on CPU;
  - int8 with 3 extra calibration sets;
  - 5 natural corruptions.

**Stronger adaptive attackers.** All use ε = 4/255:

| Attacker | Steps | Gradient through JPEG |
|---|---|---|
| BPDA-10 | 10 | Identity (round 2) |
| Surrogate-10 | 10 | Differentiable JPEG (Shin & Song 2017 rounding approximation) |
| Surrogate-50 | 50 | Differentiable JPEG |
| BPDA-50 | 50 | Identity (JPEG-75 only, completeness) |

- The differentiable JPEG (`hiftstress/torchjpeg.py`) matches OpenCV's output to about 46 dB PSNR, which is under one grey level.
- **Worst case per video:** for every video we take the lowest score over all attacks, as if the attacker picked the best attack for each video. The undefended baseline is treated the same way, using the lowest of PGD-10 and PGD-50.

## 3. Pre-registered rules

| Rule | Test | Pass |
|---|---|---|
| **H6** robustness holds | Worst-case success with the selected JPEG minus worst-case undefended (TEST) | ≥ +0.10, and 95% interval above 0 |
| **H7** clean cost | Selected JPEG, clean, minus no defence (TEST) | ≥ −0.010 |
| **H8** time overhead | Defence time divided by one tracking step without the defence, measured on the same device | ≤ 3.85%. GPU: OpenCV or GPU JPEG on the A40. CPU: JPEG next to int8 at 4 threads |
| **H9** int8 accuracy | Mean change vs fp32 CPU over 4 calibration sets (round 2 plus 3 new) | ≥ −0.010 (SD reported) |
| **H10** combined, CPU | int8 + JPEG: (a) clean vs fp32 CPU; (b) fp32 network / (int8 network + JPEG); (c) worst of BPDA-10 and surrogate-10 vs int8 undefended | (a) ≥ −0.010; (b) ≥ 1.5×; (c) ≥ +0.10 with interval above 0 |

**Claims these support:**
- **Claim R (robust HiFT):** H6 + H7 + H8 on the GPU.
- **Claim E+R (efficient and robust HiFT on CPU-class hardware):** H6 + H9 + H10 + H8 on the CPU.

If neither claim holds, the report states which rule failed and by how much. A filter that collapses under the surrogate attack is itself a finding, consistent with Athalye et al. (2018).

## 4. Will not claim

- **Drone hardware speed.** The paper's 31.2 FPS is on the Jetson Xavier's GPU; ours is A40 and Xeon.
- **Physical-world attacks.**
- **Robustness beyond L∞ ε = 4 white-box pixel attacks.**
- **Any result chosen after looking at TEST.**
- **Corruption results as a pass/fail claim.** They are descriptive: crop-level ImageNet-C severity 3, not the full UAV-C protocol.

## 5. Known confounds and how they are handled

| Issue | Handling |
|---|---|
| Quality chosen on DEV, which has only 20 videos | The rule is fixed in advance. TEST clean cost for every quality is also reported, but not used to choose |
| The surrogate is not bit-identical to OpenCV | 46 dB PSNR agreement. The real defence is always OpenCV at test time; the surrogate only supplies gradients |
| Mixed GPU types (queue 07 in round 2 ran on an L40S) | Timing rules use in-run timing divided by a baseline from the same device. The benchmarks record the hardware |
| int8 robustness | The attacker uses fp32 weights, because int8 operations have no gradients. This is stated as the threat model |
