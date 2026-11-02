# Review of the matched re-timing (2 Nov 2026) and paper changes

## Verdict
- **Accepted.** Both jobs ran on TC2N01, a Xeon Gold 6230R (the same CPU model as all earlier CPU runs), with nothing running alongside them.
- I recomputed the medians and ranges from `sessions.csv`; they match the reports.
- The model timed is the accuracy-tested one:
  - calibration videos are identical (both jobs);
  - at 4 threads, boxes matched the saved accuracy runs on 100% of frames, for both fp32 and int8.

## Headline numbers (Xeon Gold 6230R, median of within-session ratios, range over sessions)

| | 4 threads (6 sessions) | 1 thread (4 sessions) |
|---|---|---|
| Network (backbone + head) | 89.1 → 50.9 ms, **1.75×** (1.74–1.76) | 170.2 → 94.1 ms, **1.81×** (1.80–1.82) |
| Tracker step (no image loading) | 90.5 → 52.5 ms, **1.72×** (1.71–1.73) | 171.9 → 95.0 ms, **1.81×** (1.79–1.83) |
| Backbone | 68.1 → 29.3 ms, 2.33× | 139.0 → 60.6 ms, 2.29× |
| Head (fp32) | 20.9 / 21.6 ms; 42% of int8 network time | 31.2 / 33.4 ms; 36% |
| Model file | 44.3 → 33.2 MB (−25%) | same |

- **Amdahl ceiling at 4 threads:** 89.1 / 21.6 ≈ 4.1×, with the head held fixed.
- **Why 1 thread gains more:** the int8 backbone gains relatively more there, and the head is a smaller share (36% vs 42%).

## The 1-thread box divergence: harmless, explained, and reported
- **What happened:** at 1 thread, tracked boxes are not bit-identical to the saved runs (which were made at 4 threads).
  - Share of frames with IoU > 0.99: fp32 bike1 0.47; int8 bike1 0.93, person11 0.58, person7_2 0.37. Other videos 1.00.
  - Median IoU ≥ 0.983 everywhere. Maximum drift 21 px (fp32) and 4 px (int8).
  - **Per-video success AUC differs by at most 0.0002.**
- **Why:** the CPU math libraries split each convolution's sums differently across 1 vs 4 threads, and floating-point addition depends on order. These last-digit differences feed back through the tracking loop (each frame's crop is centred on the previous box), so trajectories drift slightly.
- **Evidence it is not a model problem:**
  1. It is fully deterministic: identical numbers in both 1-thread sessions.
  2. **fp32 shows it too.** fp32 has no calibration, so this cannot be a different int8 model.
  3. At 4 threads the same build reproduces the saved boxes exactly.
  4. Calibration identity passed in both jobs.
- **Effect on speed: none.** Every frame processes one fixed-size 287×287 crop, so cost does not depend on box position. 1-thread step time (95.0 ms) ≈ network (94.1 ms) + small overhead, as expected.
- **What we must not say:** that 1-thread accuracy is identical to the 4-thread experiment. Accuracy claims stay with the 4-thread runs.

## Exact changes for the paper (Codex)

1. **Abstract and conclusion: replace the speed sentence with:**
   > "On an Intel Xeon Gold 6230R CPU with four PyTorch threads, in six matched sessions against the exact accuracy-tested model, int8bb-s1 reduced network latency from 89.1 to 50.9 ms (1.75×, range 1.74–1.76×) and the per-frame tracker step from 90.5 to 52.5 ms (1.72×, 1.71–1.73×), excluding image loading. With one thread, both speed-ups were 1.81×. The serialized model shrank from 44.3 to 33.2 MB (about 25%)."
2. **Results timing paragraph:**
   - Replace the 94.4 / 52.3 ms, 1.80× / 1.76× numbers and the "historical sessions" discussion with the table above.
   - Keep one sentence of history, if space allows: "An earlier, unmatched benchmark with a differently calibrated model gave 1.80×."
3. **Amdahl paragraph:**
   - Backbone 68.1 → 29.3 ms; head 20.9 / 21.6 ms (42% of int8 network time); ceiling ≈ 4.1× (was 3.9×).
   - Keep the caveat that the head is assumed fixed.
4. **Markers:**
   - **"matched-session timing of the exact selected calibrated tracking model":** resolved. "Calibration videos identical to the accuracy run; at four threads, tracked boxes identical on all checked frames."
   - **"independent matched timing sessions with retained inputs, samples and spread":** resolved. 6 + 4 sessions, alternating order; per-session values in `sessions.csv`; timing videos named.
   - **"same-job tracker timing with identical instrumentation and matched image-inclusive runtime":** resolved for same-job tracker timing. Replace "image-inclusive" with an explicit statement that image loading is excluded by design.
   - **"benchmark crop/pixel hashes, configuration snapshot, process affinity and CPU load/frequency records":** reword to what is recorded: node, CPU model, torch version, thread count, job IDs, timing videos and frames, session order, exclusive use (no other jobs of ours). Say that CPU frequency and affinity were not controlled.
   - **"original checkpoint … byte-level verification":** the checkpoint SHA-256 (8c697d2b…) was re-verified on 2 Nov and matches the recorded hash. The HiFT commit and diff check comes with the layer-sensitivity package.
   - **Per-layer sensitivity:** pending; the layer-sensitivity package arrives tonight.
5. **One-thread identity, one sentence in setup or appendix:**
   > "One-thread runs reproduced the four-thread trajectories up to floating-point drift from thread-dependent summation order (median IoU ≥ 0.98; per-video success AUC within 0.0002); fp32 drifted likewise, and timing is unaffected because every frame processes a fixed-size crop."
6. **Figure F16:** use `F16_method_diagram.png` (no numbers) in the paper. The footnoted version now carries the matched numbers.
7. **Everywhere:** "1.80×" → "1.75×" (network) and "1.76×" → "1.72×" (tracker step), unless explicitly labelled as the earlier unmatched benchmark.
