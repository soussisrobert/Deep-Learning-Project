# Round 5 plan: failure/attack awareness, natural corruptions, and closing round 4

Owner: Manzi (adversarial + edge tracks). **Status: prepared, not run.**

The rules below are fixed before any round-5 result exists. `scripts/r5_gate.py` applies them, cuts losing branches, and lists which queues may run next.

- **Queues:** `plan/round5/SCHEDULE.md`.
- **How to run them:** `AGENT_RUNBOOK_ROUND5.md`.

## Part R4C: close round 4 (CPU only, no gate)

The round-4 analysis named two open doubts about the int8 claim:
- the TEST set has been reused across rounds;
- there is no full-benchmark (123-video) number.

| Step | What runs |
|---|---|
| **R4C2** | The selected recipe (int8bb-s1, 40 videos × 4 spread calibration frames) on the **83 non-TEST videos**, calibrated on the 40 TEST videos. Roles are swapped, so calibration and evaluation never share a video. Plus 2 sensitivity draws with 30 calibration videos |
| **R4C1** | Exports the official UAV123 attribute labels (UAV.json) for a per-attribute breakdown with real labels instead of proxies |

**Claim E-full:**
- **Pooled across 123 videos:** the 40 TEST videos contribute the mean of the 3 round-4 draws, and the 83 others the exact recipe.
- **Rule:** mean change vs fp32 CPU ≥ −0.010.
- **Reported alongside:** the interval, and the attribute table. The attribute table is descriptive.

## Part F/A: can HiFT tell when it has failed, or is being attacked?

**Signal.** HiFT's own confidence for the box it picks (`best_score`; map peak and PSR are reported as secondary signals). Every new run saves it per frame (`results/<run>/scores/`).

**Definitions (fixed):**
- **Failure:** overlap with the ground truth below 0.1. Frames where the target is absent are excluded.
- **Alarm:** confidence below the 5th percentile of clean DEV frames. The threshold is fixed once from DEV in `plan/round5/threshold.json`, so about 5% false alarms are expected.
- **Adaptive attacker (`--objective shift`):** pushes the target's score down *and* keeps the best score elsewhere high, so the tracker jumps without its confidence dropping. λ weights the keep-confidence-high term.

**Gates.** TEST = the fixed 40 videos; DEV = the 20 round-3 DEV videos.

| Gate | Test | Pass, else CUT |
|---|---|---|
| **P0 / F0** | DEV clean: failure-warning AUROC (confidence ranks failed below good frames) | ≥ 0.70, else the F branch is cut |
| **P0 / A0** | DEV standard PGD-10 (ε = 4): share of attack-broken frames that raise the alarm | ≥ 80%, else the A branch is cut, including the adaptive pilot |
| **P1** | DEV, λ ∈ {1, 3, 10}: choose the **strongest evader** among attackers that still work (DEV success drops by ≥ 0.20); if none work, the most damaging λ | Choice written to `attacker_selected.json` |
| **Claim F** | TEST clean failure-warning AUROC | ≥ 0.75, with the 95% interval (over videos) above 0.5 |
| **Claim A** | TEST alarm on attack-broken frames: standard PGD-10 ≥ 80% **and** adaptive (chosen λ) ≥ 50%; clean TEST false alarms ≤ 10% | All three |

## Part K: natural corruptions (descriptive, plus one rule)

**Corruptions** (all on TEST):
- ImageNet-C-style, crop level, severity 3: Gaussian noise, Gaussian blur, motion blur, contrast, low light.
- Low light at severity 5.
- **Synthetic night** (severity 3): darker + sensor noise. **This is not real night footage.**

**Runs:** fp32 with and without JPEG-75 (GPU); int8bb-s1 (CPU, calibration draw 1).

**Claim K:** int8 holds up as well as fp32 under corruption. Test: mean over the 7 corruptions of (int8 drop − fp32 drop) ≥ −0.010. Each drop is measured against its own clean run on the same device.

The failure-warning AUROC under each corruption is reported (descriptive).

## Will not claim

- Real night performance: there is no night dataset.
- Physical attacks.
- Drone hardware.
- Detection against attackers other than the two tested (standard and confidence-preserving PGD-10, ε = 4).
- Anything chosen after looking at TEST.
