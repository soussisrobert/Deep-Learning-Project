# Round 4 plan: try to turn the two near-misses into passes (gated)

Owner: Manzi (adversarial + edge tracks). **Status: prepared, not run.**

The rules below are fixed before any round-4 result exists. `scripts/r4_gate.py` applies them. It decides which queue may run next, and it cuts a branch the moment that branch fails a gate.

- **Thresholds:** unchanged from rounds 2–3.
- **Queues:** `plan/round4/SCHEDULE.md`.
- **How to run them:** `AGENT_RUNBOOK_ROUND4.md`.

## 1. Why these tests

| Round-3 result | What round 4 tries |
|---|---|
| JPEG-75 is robust (worst-case gain +0.151, H6 pass) but its TEST clean cost is −0.0116 (H7 limit −0.010) | **JPEG-85.** On the round-3 DEV videos it was nearly as robust (0.274 vs 0.289 under surrogate-10; gain +0.151) with an almost-zero, tight clean change (+0.004 [−0.001, +0.013]) |
| int8 backbone accuracy depends on the calibration draw (−0.005 to −0.025; mean −0.016, H9 fail) | **Two fixes, tested first on DEV:** more and better-spread calibration frames, and keeping the first conv in fp32 |

**Disclosure.** JPEG-85 is chosen *after* seeing JPEG-75 fail H7 on TEST. The choice uses only round-3 DEV evidence: it was the runner-up on the DEV selection rule, with the tightest clean interval. TEST has never been used for JPEG-85, and round 4 uses it exactly once per test.

**Local signal before any GPU time** (real HiFT weights, natural photos, PyTorch 2.8):

| Variant | Score-map error vs fp32 | Peak agreement | CPU speed-up (4 threads) |
|---|---|---|---|
| int8bb | 0.028 | 0.84 | 2.2× |
| int8bb-s1 (first conv fp32) | 0.016 | 0.89 | 1.65× |
| int8bb-s15 (first and last conv fp32) | No better than s1, and slower | | |

More calibration crops narrowed int8bb's draw-to-draw error range (0.025–0.032 → 0.023–0.025). The local signal is only a reason to test on DEV, not evidence for a claim.

## 2. Branches and gates

TEST = the fixed 40 videos (`attack-subset-40.json`). DEV = the 20 round-3 DEV videos. All attacks use ε = 4/255.

### J: JPEG-85 (GPU, plus one CPU timing job)

| Gate | Test | Pass, else the J branch is CUT |
|---|---|---|
| **J1 = H7** | TEST clean change vs no defence (GPU) | ≥ −0.010 |
| J1c (runs alongside) | fp32 on CPU, 4 threads: clean with and without JPEG-85 on TEST. Gives **H8-CPU**: JPEG time / tracking time | ≤ 3.85% |
| **J2** | Surrogate-10 gain vs undefended PGD-10 | ≥ +0.10 and interval above 0 (without this, H6 cannot pass) |
| **J3 = H6** | Worst case per video over surrogate-10, surrogate-50 and BPDA-10, against the worst undefended (PGD-10 / PGD-50) | ≥ +0.10 and interval above 0 |

### Q: int8 recipes (CPU)

**Q0, the DEV pilot.** Calibration never uses TEST videos. Four recipes, 3 random calibration draws each:

| Recipe | Variant | Calibration |
|---|---|---|
| A | int8bb | 20 videos × 2 frames (round 3) |
| B | int8bb | 40 videos × 4 frames spread through each video |
| C | int8bb-s1 | 20 × 2 |
| D | int8bb-s1 | 40 × 4 spread |

**Eligible on DEV** if all three hold:
- the mean change vs fp32 CPU is ≥ −0.010;
- the worst draw is ≥ −0.020;
- the network speed-up at 4 threads, from `results/edge_round4`, is ≥ 1.5×.

**Choice:**
- Take the best mean; within 0.002 of the best, take the faster recipe (speed-ups that round to the same tenth count as equal, and then the more accurate recipe wins).
- If none is eligible, the Q and C branches are CUT.

**Q1 = H9'.** The chosen recipe on TEST with 3 new calibration draws. Pass: mean change vs fp32 CPU ≥ −0.010, and speed-up ≥ 1.5×.

### C: chosen int8 recipe + JPEG-85 on CPU (only if J1 and Q1 pass)

| Gate | Pass |
|---|---|
| **C1** | Clean change vs fp32 CPU ≥ −0.010, and speed-up including JPEG time ≥ 1.5× |
| **C2** | Worst of BPDA-10 and surrogate-10 vs the same int8 model undefended (attacker uses fp32 weights): gain ≥ +0.10 with interval above 0 |

## 3. Claims

| Claim | Requires |
|---|---|
| **R-CPU** (robust HiFT on CPU-class hardware) | J1 + J2 + J3 + H8-CPU |
| **E** (efficient HiFT) | Q0 + Q1 |
| **E+R** | Everything above + C1 + C2 |

GPU overhead is still reported; round 3 established that it fails on an A40.

## 4. Will not claim

- Drone-hardware speed (a Xeon is not a Jetson).
- Physical-world attacks.
- Results from DEV pilot runs. These only choose what goes to TEST.
- Anything about qualities, recipes or attacks that were not run.
