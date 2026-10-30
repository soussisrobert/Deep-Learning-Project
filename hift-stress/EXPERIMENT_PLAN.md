# Round 2 plan: turning "HiFT is fragile" into a tested improvement

Owner: Manzi (adversarial + edge tracks). Status: **prepared, not run.** The thresholds below are fixed
*before* any round-2 result exists. `scripts/evaluate_round2.py` applies them mechanically.

## 1. What counts as an improvement for *this* paper

HiFT's stated aim is "an efficient and effective hierarchical feature transformer for aerial tracking".
It chose a lightweight AlexNet backbone because aerial platforms cannot afford deeper networks, and it
backs the aim with two numbers:
- **Effective:** UAV123 success **0.589**, precision **0.787**. This is first place, **+0.010 success** over SiamRPN++ (0.579).
- **Efficient:** about 130 FPS on a desktop GPU, and **31.2 FPS onboard an NVIDIA AGX Xavier** without TensorRT.
  That is only just above the 30 FPS of UAV123's video.

(Cao et al., ICCV 2021, [arXiv:2108.00202](https://arxiv.org/abs/2108.00202), Table 1 and Sec. 4.)

**So an improvement must move one of these two axes without breaking the other.** We test two kinds:

| Axis | Proposed change | Improvement means | Yardstick taken from the paper |
|---|---|---|---|
| Effective, under attack (a realistic threat HiFT never tested) | Training-free input defence in front of HiFT | Regain success under an attacker who **knows** the defence, while keeping clean success and real-time speed | Clean cost ≤ **0.010** AUC (HiFT's whole lead over SiamRPN++). Time overhead ≤ **3.85%** (1 − 30/31.2: HiFT's onboard real-time margin) |
| Efficient | Post-training int8 quantisation (backbone; backbone + head convolutions) | Faster with no meaningful loss of success | Speed-up ≥ **1.5×**. Accuracy drop ≤ **0.010** AUC (same yardstick) |

## 2. Where this sits in published work (numbers we compare against)

| Work | Setting | Reported numbers | What it tells us |
|---|---|---|---|
| **Our round 1** (TC2, 17 Oct) | White-box PGD-10, ε=4/255 on HiFT's search crop, UAV123 40-sequence subset | Success 0.582 → **0.095** (−84%). Same-size noise: 0.563 | Baseline to improve on |
| Ad²Attack (Fu et al., ICRA 2022) | Online attack on 7 UAV trackers incl. **HiFT**; UAV123, UAVDT, UAVTrack112 | HiFT precision 0.723 → 0.391 (−45.9%), success 0.543 → 0.263 (−51.5%), averaged over benchmarks | An independent group also finds HiFT breaks under small perturbations. Our white-box PGD is stronger |
| Cooling-Shrinking Attack (Yan et al., CVPR 2020) | Generator that "cools" the target heat-map, against SiamRPN++ | Fools SiamRPN++ and transfers to DaSiamRPN and DiMP | Our attack objective follows the same idea |
| UAV-C corruption benchmark ([arXiv:2403.11424](https://arxiv.org/abs/2403.11424)) | 18 natural corruptions; UAV123@10fps + DTB70 | HiFT clean 57.3 → Gaussian noise 46.1, Gaussian blur 49.9, average over corruptions 43.4 | Natural noise and blur already cost HiFT 10–20%. A blur or median **defence** must not add a similar clean loss |
| Pluggable U-Net purifier ([par.nsf.gov 10633008](https://par.nsf.gov/servlets/purl/10633008); ICTAI 2024 preliminary) | Trained defence, SiamRPN++, UAV123 | Ad²Attack: 0.373 → **0.555** success (clean 0.597). Jetson Xavier speed 40 → 12 FPS (**+233% compute**) | Strong recovery is possible, but trained purifiers break real-time. This motivates training-free defences |
| CMDN (Drones 2024, [mdpi.com/2504-446X/8/11/607](https://www.mdpi.com/2504-446X/8/11/607)) | Trained defence, transfers to **HiFT**; adaptive attacks tested | Clean success cost up to 7.3%. Adds 8.8 ms/frame on an RTX 3090. 27 FPS on Jetson Orin | Same trade-off: robustness bought with clean accuracy and time |
| LRR (ICLR 2024, [arXiv:2404.06247](https://arxiv.org/abs/2404.06247)) | SiamRPN++ on OTB100 | JPEG q=90 raises RTAA-attacked precision 32.7 → 66.1. Resizing 32.7 → 49.3. DiffPure costs 146–3391 ms/frame | Cheap input transforms do help against **non-adaptive** attacks. Our test adds the adaptive case |
| Athalye et al., ICML 2018; Carlini et al. 2019 ([arXiv:1902.06705](https://arxiv.org/abs/1902.06705)) | Evaluation methodology | Input-transform defences often collapse under adaptive (BPDA/EOT) attacks | Why the adaptive number is the one we judge on |
| Jacob et al., CVPR 2018 ([arXiv:1712.05877](https://arxiv.org/abs/1712.05877)) | int8 integer-only inference | ImageNet top-1 within about 2%; detection about −1.8% relative; latency about 1.9–2.2× faster on mobile CPUs | Basis for the 1.5× / 0.010 thresholds. Post-training int8 can hurt small models, so the accuracy check matters |

Our contribution on top of these:
- The first **adaptive**, paired, per-sequence evaluation of training-free defences **on HiFT**, judged against HiFT's own real-time margin.
- A measured int8 speed/accuracy trade-off for HiFT on CPU, including whether compression changes attack robustness.

## 3. Hypotheses and pre-registered pass rules

All on the fixed 40 UAV123 sequences (`attack-subset-40.json`) at ε = 4/255, PGD 10 steps. Paired by sequence, with a 95% bootstrap over sequences.

- **H1 (defence works when the attacker is unaware).** Success under PGD with defence d is above PGD without it (interval above 0).
- **H2 (defence survives an adaptive attacker): the headline.**
  - Adaptive gain ≥ +0.10 success with interval above 0.
  - Clean cost ≥ −0.010.
  - Tracking-time overhead ≤ 3.85%.
  - All three hold → **"improvement"**. Otherwise we report the trade-off honestly.
- **H3 (int8 is a free speed-up).**
  - Network ≥ 1.5× faster at 4 CPU threads.
  - Success drop ≤ 0.010 vs fp32 on the same CPU, on 40 sequences, then confirmed on all 123.
- **H4 (compression and robustness).** PGD success of int8 vs fp32. The attacker uses fp32 weights: the realistic case, where the attacker has the published model. Reported in both directions; no claim beyond the measured gap.
- **H5 (the headline attack is stable).** PGD ε=4 over seeds 0/1/2 has SD ≤ 0.02. If not, every attack number is reported as mean ± SD.

**Things we will not claim:**
- Physical-world or real-drone robustness.
- Significance from frames.
- Any result from smoke or probe runs.
- Onboard FPS. A Xeon is not a Xavier; we report relative speed-up and our own hardware only.

## 4. Known confounds and how they are handled

| Issue | Handling |
|---|---|
| Our precision rule differs from HiFT's toolkit (absent target = miss vs hit) | Both are now computed (`precision_20_official`). Success is unaffected. Old runs are re-scored by `scripts/rescore.py` (job 0) |
| int8 calibration on test data (round 1 used overlapping sequences) | Calibration now excludes the evaluated sequences by default (20 other UAV123 sequences, 2 frames each). For the 123-sequence run some overlap is unavoidable, and it is disclosed |
| CPU vs GPU numerics move single sequences by up to 0.19 | Compare only within one device. Claims rest on the 40-sequence mean with an interval |
| JPEG and bit-depth are not differentiable | Adaptive attack uses BPDA (straight-through). Blur and median use exact gradients |
| Attack time (151 ms/frame) | Reported separately. The attack is an offline upper bound, not a claim about real-time attackers |

## 5. Schedule (from `plan/SCHEDULE.md`; times from measured round-1 costs)

- **Job 0:** `sbatch slurm/job_rescore_tc2.sbatch` (CPU, under 1.5 h). Re-scores every finished run with both precision rules and per UAV123 attribute, and benchmarks int8full.
- **P1, defences:** queues 01–04, GPU, about 16 h in total. **This is the core result.**
- **P2, int8 accuracy on 40 sequences:** queue 05, CPU, about 2.3 h. Run it next to a GPU job.
- **P3, attack curve + seeds:** queues 06–08, GPU, about 9 h.
- **P4, compressed models under attack:** queue 09, GPU + CPU, about 4.6 h. The attack runs on the GPU and the int8 model on the CPU, which avoids about 21 h of CPU-only attack.
- **P5, int8 on all 123 sequences:** queue 10, CPU, about 3.6 h.

TC2 allows two jobs, 10 CPUs, 30 GB and one GPU in total. The GPU queues (6 CPUs / 20 GB) pair with the CPU queues (4 CPUs / 8 GB). Minimum about 30 GPU-hours if every queue runs; if time is short, run P1 + P2 first.

After each batch, on TC2:

```bash
python scripts/evaluate_round2.py    # writes results/round2/criteria.md: one verdict row per defence and per variant
```

## 6. After the verdicts (only if H2 or H3 passes)

- Combine the best defence with int8full on the CPU and run clean + adaptive PGD.
  - If both still pass, that is the "efficient and robust HiFT" configuration for the report.
  - If no defence passes H2, the report says so. Weak adaptive results are themselves the finding (consistent with Athalye et al.). The next step would be adversarial fine-tuning, which needs HiFT's training sets and is out of scope.
