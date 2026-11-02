# Review: per-layer int8 sensitivity (3 Nov 2026)

## Verdict: accepted
- **Jobs:** all three completed on TC2N01 (Xeon Gold 6230R).
- **Identity:** calibration videos matched round 4 for all three draws, and the re-tracked `none`/`keep1` boxes matched round 4 on 100% of frames.
- **Completeness:** all 12 new DEV runs have 20 videos and 16,826 frames.
- **Fidelity sample:** Stage A used **192** valid DEV template/search pairs. The plan said 200; 8 sampled frames had invalid or absent ground-truth boxes. Report 192.

## Q1 (primary, pre-registered): which backbone layer is most sensitive to int8?

**Layer 1.** Measured as the relative L2 error of the fused score map when only that layer is int8 (mean over 3 draws):

| Layer | layer1 | layer2 | layer3 | layer5 | layer4 |
|---|---|---|---|---|---|
| Error | **0.0130** | 0.0082 | 0.0037 | 0.0036 | 0.0022 |

**Robustness checks:**
- **Same ranking in every draw.** Layer 1 error is 0.0128 / 0.0135 / 0.0129 for draws 1–3.
- **Seen from the other direction too:** keeping layer 1 in fp32 (`keep1`) gives the lowest score-map error of the five single-layer choices in every draw (mean 0.0091, vs 0.0139–0.0148 for keep2–5). It also gives the highest same-peak agreement (0.976, vs 0.948–0.957).

## Q2 (secondary, descriptive): is keeping layer 1 best for DEV tracking accuracy?

**Not established.** DEV success AUC change vs fp32 CPU, per-video mean over 3 draws, 95% paired bootstrap:

| fp32 layer | All 20 videos | Excluding person19_2 (pre-specified) |
|---|---|---|
| none (all int8) | +0.0197 [−0.0102, +0.0692] | −0.0031 [−0.0129, +0.0030] |
| keep1 | +0.0251 [−0.0035, +0.0738] | +0.0024 [−0.0057, +0.0110] |
| keep2 | +0.0464 [−0.0057, +0.1194] | +0.0250 [−0.0099, +0.0833] |
| keep3 | +0.0213 [−0.0051, +0.0585] | +0.0064 [−0.0081, +0.0276] |
| keep4 | +0.0307 [−0.0032, +0.0793] | +0.0164 [−0.0048, +0.0543] |
| keep5 | +0.0206 [−0.0069, +0.0578] | +0.0058 [−0.0111, +0.0279] |

- **Ranking:** layer 2 has the highest mean.
- **Uncertainty:** every interval includes zero and they all overlap. DEV (20 videos) cannot distinguish the placements, as the plan anticipated.
- **No changes:** the selected recipe (int8bb-s1) is unchanged, and nothing was run on TEST.

## Speed of each placement (descriptive; Stage A ran alone, 4 threads)

| Config | none | keep1 | keep2 | keep3 | keep4 | keep5 |
|---|---|---|---|---|---|---|
| Network | 40.6 ms | 49.6 ms | 54.7 ms | 46.1 ms | 47.9 ms | 44.9 ms |

fp32 network: 89.0 / 86.7 ms (start / end).

Keeping layer 2 in fp32 is the slowest single-layer choice. Layer 2 is the most expensive convolution, which is why its int8 conversion matters most for speed.

## Paper wording (replaces the per-layer sensitivity limitation)

> "A per-layer sweep on the 20 DEV videos confirmed the heuristic: quantising `layer1` alone distorted HiFT's fused score map most (relative L2 error 0.013, versus 0.002–0.008 for layers 2–5; same ranking in all three calibration draws), and keeping `layer1` in fp32 gave the most faithful score maps of the five single-layer choices. DEV tracking accuracy did not separate the placements: all changes versus fp32 had overlapping 95% intervals that included zero, so we make no claim that `layer1` is the best placement for tracking accuracy."

## HiFT code pin, cluster confirmation (clears the last marker)

The TC2 checkout gives, verbatim (`cluster_hift_git_check.txt`):
```
992aa7c1605360d607af84cb804b93cd8ee81caf
 pysot/models/model_builder.py | 4 ++--
 pysot/models/utile/utile.py   | 4 ++--
 2 files changed, 4 insertions(+), 4 deletions(-)
```
- **Match:** the same HEAD, and the same 2 files and 4 changed lines as the workspace checkout (`../2026-11-02_hift-pin-check/PIN_CHECK.md`). No `.pyc` deletions on the cluster.
- **Line content:** the agent saved the stat but not the full diff text. The line contents are pinned by `tools/patch_hift.py`, which refuses to run unless each of its 4 exact edits matches exactly once.
