# Round 5 results: final analysis (22 Oct 2026)

Owner: Sandhy (adversarial + edge tracks).

**Evidence:**
- The round-5 handoff archive. All 5,948 SHA-256 hashes match.
- The deployed `source/` is identical to the repo.
- Every number below was recomputed from the saved boxes, per-frame confidence files and the supplied ground truth. The script is `analyse5.py`; its full output is in `analyse5_output.txt` and `numbers5.json`. The gate's own scorecard is copied in `criteria_copy.md`.
- Recomputation agrees with the gate on every claim. No problems were found: row counts match and the video sets are correct.
- **No rule, threshold or video list was changed.**

## Team summary (plain language)

1. **The compressed model is a real speed win at a small accuracy cost.**
   - **Speed:** int8bb-s1 runs the network 1.8× faster on a 4-thread CPU (94 → 52 ms), and the model shrinks from 44 to 33 MB.
   - **Accuracy over all 123 videos:** the change is +0.003, which is statistically zero, and 114 of 123 videos move by less than 0.02.
   - **But the average is flattered:** two fragile videos happened to flip in its favour. Without them, the honest cost is about **−0.004** (interval −0.008 to −0.001), still well inside our −0.010 limit.
   - **A few videos lose a lot:** uav2 −0.16, car4_s −0.11, group1_2 −0.07.
2. **HiFT's own confidence score is a good *after-the-fact* failure detector, not an early warning.**
   - **Ranking:** it ranks failed frames below good ones very well (AUROC 0.97 on TEST). This also holds *inside* individual videos (median 0.975), so it is not just separating easy videos from hard ones.
   - **When the alarm fires, it is almost always right:** 98% precision, and only 0.2% of well-tracked frames raise it.
   - **But it misses most failures:** it catches 42% of failed frames, and it was already on just before a failure started in only 10 of 32 cases.
3. **Confidence cannot detect an attack.**
   - Under PGD, HiFT loses the target while staying as confident as when it tracks correctly (fig 11). Only 0.3% of attack-broken frames raised the alarm.
   - Flipping the rule ("too confident = suspicious") doesn't help either: AUROC 0.41.
   - So the attack-alarm branch was correctly cut at the pilot, and nothing was run on TEST.
4. **The compressed model handles corrupted video about as well as the original.**
   - Across 7 corruptions (noise, blur, motion blur, contrast, two darkness levels, synthetic night), its extra loss averages −0.0035.
   - No single corruption shows a clear difference. Noise (−0.021) and very dark (−0.017) lean worse, but their intervals include zero.
   - Confidence still detects failures under corruption (AUROC 0.89–0.98, the same for both models).
   - The JPEG-75 filter neither helps nor hurts much under corruption (−0.019 to +0.008).

## What we can now claim (and not claim)

| Claim | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| **E-full:** int8bb-s1 keeps accuracy on all 123 videos | mean change ≥ −0.010 | +0.0030 [−0.0054, +0.0140]. Without the two flipped videos: −0.0037 [−0.0076, −0.0006] | **Supported** (report both numbers) |
| **F:** confidence predicts natural failure | TEST AUROC ≥ 0.75, interval above 0.5 | 0.968 [0.938, 0.989]. DEV 0.920 | **Supported** |
| **A:** confidence flags attacks | DEV pilot ≥ 80% of broken frames flagged | 0.3% of 11,303 frames | **Failed at the pilot.** Not run on TEST |
| **K:** int8 as robust as fp32 under corruption | mean extra drop ≥ −0.010 | −0.0035 | **Supported** |

**Not claimed:**
- Real night: the night test here is synthetic darkening plus noise.
- Physical attacks, drone hardware, or attackers other than PGD-10 at ε = 4.
- That confidence gives *early* warning.
- Any per-video or per-attribute difference: none is significant.

## Technical appendix

### A1. Data and provenance
- **TEST:** the fixed 40 videos (37,681 frames). **DEV:** 20 videos (16,826). **Non-TEST:** 83 videos (74,897). Together, TEST and non-TEST cover all 123 UAV123 videos.
- **E-full pooling** (as pre-registered):
  - The TEST videos use the mean of the 3 round-4 calibration draws.
  - The 83 non-TEST videos use the exact recipe (int8bb-s1, 40 TEST videos × 4 spread calibration frames), so calibration and evaluation never share a video.
  - Each is compared against fp32 on the same CPU with 4 threads.
- **Noise floor:** fp32 CPU vs fp32 GPU differs by +0.0027 [−0.0007, +0.0071] on the same 123 videos. Changes smaller than about ±0.005 are within run-to-run numerical noise.
- **Intervals:** all intervals are 95% paired bootstraps over videos (10,000 resamples; 2,000 for AUROC).

### A2. Efficiency (claim E-full)
- **Network time** (4 Xeon threads): fp32 94.4 ms → int8bb-s1 52.3 ms (1.80×). The full tracker step on the 83-video run is 53.5 ms per frame.
- **Why not more than about 1.8×:**
  - Only the AlexNet-style backbone is int8; layer1 stays fp32 because quantising it costs accuracy.
  - The transformer head stays fp32 and is now most of the run time.
  - By Amdahl's law, even an infinitely fast backbone gives about 3.9×.
  - Larger gains need int8 or fused kernels for the head, for example OpenVINO or ONNX Runtime. That was not tested.
- **Accuracy split:**

  | Set | Change in success AUC |
  |---|---|
  | TEST 40 | −0.0074 [−0.0168, −0.0013] |
  | Non-TEST 83 | +0.0080 [−0.0033, +0.0239] |
  | Sensitivity draws, 30 calibration videos | +0.0078 and +0.0022 |

- **Distribution:** the median change is +0.0001; 114/123 videos are within ±0.02; 2 rise by more than 0.05 (person19_2 +0.456, uav4 +0.359) and 3 fall by more than 0.05 (uav2 −0.155, car4_s −0.110, group1_2 −0.073).
- **Why the two rises are flips, not improvements:** they are recover-or-lose-the-target videos where a tiny numeric difference changes the outcome. Round 3 already saw person19_2 swing the DEV selection. With them excluded the change is −0.0037 [−0.0076, −0.0006]. **We recommend quoting this as the cost.**

### A3. Failure awareness (claim F)
- **Signal and definitions:** the signal is HiFT's `best_score` for the chosen box. Failure means IoU < 0.1. Target-absent frames and frame 0 are excluded.
- **Frame counts on TEST:** 6,075 failed / 31,020 good. The failures are concentrated: 17 of 40 videos have any, and five videos (person1_s, person13, car2, uav1_1, group1_4) hold 60% of them. That is why we also report the within-video AUROC: the median is 0.975 over the 13 videos with at least 20 failed and 20 good frames.
- **The fixed alarm** (DEV clean 5th percentile, threshold 0.905):
  - It fires on 7.1% of clean TEST frames. **This is the gate's "false alarm" figure, but most of these frames are real failures.**
  - On good frames, it fires only 0.2% of the time.
  - Recall 42%, precision 97.6%.
- **Early warning:** of 32 failure onsets (a failed frame after at least 10 good ones), the alarm was on in the preceding 5 frames for only 10.
- **Under corruption** (descriptive, fp32 / int8 AUROC):

  | Corruption | fp32 | int8 |
  |---|---|---|
  | Gaussian noise | 0.955 | 0.950 |
  | Gaussian blur | 0.978 | 0.980 |
  | Motion blur | 0.937 | 0.942 |
  | Contrast | 0.892 | 0.916 |
  | Low light 3 | 0.963 | 0.960 |
  | Low light 5 | 0.946 | 0.940 |
  | Night 3 | 0.970 | 0.969 |

### A4. Why the attack alarm failed (claim A)
- **DEV, PGD-10 at ε = 4:** 11,303 frames lost the target. The median confidence of those frames is 2.235, against 2.326 for well-tracked clean frames and 1.445 for naturally failed frames.
- **Interpretation:** 39% of attack-broken frames are *more* confident than the median good frame (fig 11).
  - The attack lowers the score at the true target until a distractor location wins.
  - That distractor still produces a high, peaked response, so the map looks healthy.
  - Natural failures (blur, occlusion, out of view) instead flatten the whole map.
- **No two-sided fix:** treating unusually high confidence as suspicious gives AUROC 0.41, so it is useless too.
- **What follows:** score-map confidence cannot serve as an attack detector for HiFT. Detection would need a separate signal, for example input statistics, feature consistency between frames, or motion plausibility. That was not tested.
- **Branch status:** because A0 failed, the confidence-preserving ("shift") attacker was never needed or run. This is per plan.

### A5. Corruptions (claim K), TEST, change in success AUC

| Corruption | fp32 drop | int8 drop | int8 extra [95% CI] | JPEG-75 effect on fp32 |
|---|---|---|---|---|
| Gaussian noise 3 | −0.079 | −0.100 | −0.021 [−0.058, +0.008] | −0.014 |
| Gaussian blur 3 | −0.059 | −0.052 | +0.007 [−0.008, +0.021] | +0.008 |
| Motion blur 3 | −0.031 | −0.036 | −0.006 [−0.025, +0.015] | −0.019 |
| Contrast 3 | −0.060 | −0.044 | +0.016 [−0.005, +0.040] | −0.008 |
| Low light 3 | −0.027 | −0.033 | −0.006 [−0.032, +0.015] | −0.014 |
| Low light 5 | −0.043 | −0.060 | −0.017 [−0.044, +0.005] | +0.006 |
| Synthetic night 3 | −0.078 | −0.076 | +0.003 [−0.031, +0.033] | +0.005 |
| **Mean** | | | **−0.0035** | |

Fig 12 plots the two drop columns.

### A6. Links to the HiFT paper and AI6103 topics
- **Efficiency:** HiFT's selling point is real-time tracking on aerial platforms. We show its backbone can be post-training quantised to int8 (a 1.8× CPU network speed-up, 25% smaller) for about 0.004 AUC. We also show, via Amdahl's law, that its attention head now limits further speed-up.
- **CNN plus attention:** the split of costs, and of quantisation sensitivity, between the CNN backbone and the transformer head is a concrete hybrid-architecture lesson. Layer1 and the attention head are the sensitive or expensive parts.
- **Evaluation:** we used a pre-registered DEV/TEST protocol, gates that cut losing branches, paired per-video bootstrap intervals, a measured noise floor, and an all-123-video confirmation. We also report a sensitivity analysis showing that an average can be driven by a couple of fragile videos.
- **Robustness:** PGD at ε = 4/255 removes about 84% of HiFT's success score. JPEG input filtering helps against attacks but costs too much clean accuracy. HiFT's confidence fails as an attack detector because adversarial failures are confident.

### A7. Measured vs proposed (keep separate in the write-up)
- **Measured:** the int8 speed and accuracy, the confidence failure detector, the attack-alarm failure and the corruption parity above.
- **Proposed, not measured:** an "edge HiFT" system that runs int8bb-s1 and uses the confidence alarm to trigger re-detection or a hand-off. We have not measured whether acting on the alarm improves tracking; only the alarm's accuracy is measured.

### A8. Possible next checks (suggested, not run)
1. **OpenVINO or ONNX Runtime int8 for the whole network:** the only route to well beyond 2× speed. It would be a new round with its own clean-cost rule.
2. **Re-detection triggered by the alarm,** on DEV first, then TEST: does it turn the 42% detection into recovered frames?
3. **Real night footage** (NAT2021) to replace the synthetic night test.
4. **A temporal-consistency attack detector**, for example box jump vs motion: the confidence route is closed.

**Recommendation:** stop experiments here and write up. Options 1–2 are only worth doing if the team wants a stronger "system" story and there is time after the report is drafted.
