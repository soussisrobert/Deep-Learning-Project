# Per-layer int8 sensitivity test: pre-registered plan (2 Nov 2026, before any result)

## Why
- The paper keeps backbone `layer1` in fp32 (int8bb-s1). That choice followed a common CNN heuristic; HiFT's per-layer sensitivity was never measured.
- The round-4 DEV pilot did not show a layer1 benefit once `person19_2` was excluded.
- This test measures the per-layer sensitivity directly, so the paper can state it rather than assume it.

## What runs (CPU, `scripts/layer_sensitivity.py`; no existing code changed)

**Configurations** (head always fp32):

| Config | Layers kept fp32 |
|---|---|
| `none` | none: all 5 backbone layers int8 (= int8bb) |
| `keep1`…`keep5` | only that layer (`keep1` = int8bb-s1) |
| `only1`…`only5` | all except that layer: only that layer is int8 |

**Calibration:** identical to the round-4 DEV pilot.
- 40 videos × 4 spread frames, draws 1–3.
- The pool excludes the DEV and TEST videos.
- The script stops unless the calibration videos match round 4's saved runs, and re-tracks one DEV video to check the boxes match.

**Stage A, fidelity** (all 11 configs × 3 draws, on 200 DEV template/search pairs):
- **same-peak:** whether the score-map peak lands on the same cell as fp32;
- **relative L2 error** of the fused score map;
- **network latency** (descriptive).

**Stage B, DEV tracking** (20 videos):
- runs `keep2`–`keep5` × draws 1–3;
- `none` and `keep1` reuse round 4's identical runs;
- reports the change in success AUC vs fp32 CPU (`uav123-dev20_fp32_clean_nodef_t4`), per-video mean over draws, with a 95% paired bootstrap; also excluding `person19_2` (pre-specified, because it flipped in round 4).

**Not run:** TEST. The selected recipe is **not** changed by this test, because TEST has already been used. This is evidence for the method's justification only.

## Questions and how the answers are read (fixed now)

**Q1 (primary).** Is layer1 the most quantisation-sensitive backbone layer? Measured by Stage A relative error when quantised alone (`only<k>`), mean over draws.
- **If layer1 ranks first:** the paper says "a per-layer sweep confirmed layer1 as the most quantisation-sensitive backbone layer", citing the error values.
- **Otherwise:** the paper says "our layer1 choice followed a heuristic; the sweep shows layer *k* is more sensitive", and lists keeping layer *k* as untested future work.

**Q2 (secondary).** Is keeping layer1 the best single-layer choice for tracking accuracy on DEV? Compare the `keep<k>` changes vs fp32 in Stage B.
- Reported descriptively, with intervals.
- If intervals overlap (likely on 20 videos), say so: "DEV cannot distinguish the placements".

**Speed:** the `keep<k>` latency shows each placement's cost, e.g. keeping a large layer fp32 costs more time. This is descriptive.

**Paper rule:** whatever the result, the "[EVIDENCE NEEDED: per-layer sensitivity]" marker is replaced by the measured statement above. No claim is changed beyond what Q1/Q2 support.
