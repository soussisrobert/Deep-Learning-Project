# HiFT: trustworthy reproduction before improvement

Research implementation for UAV single-object tracking. Source-resolution-aware fusion remains a **hypothesis**, not an established contribution. **Wave 4** establishes bounded authentic temporal positives, simultaneous known-distinct negatives, actual Windows/CUDA input-path correctness and a separately gated corrected-B2 reliability-pilot proposal—not real-data SGD, a learned baseline, full benchmark reproduction or R1. See [Wave 4 A–G delivery and gates](<reports/wave4_progress.md>) and [compact evidence](<experiments/wave4_validation.json>); all prior [Wave 3](<reports/wave3_progress.md>) and [Wave 2](<reports/wave2_progress.md>) findings remain frozen.

## Domain-tracking close-out (stage 7, 2026-11-04) — REF frozen

[Stage-7 close-out](<reports/STAGE7_CLOSEOUT.md>) freezes REF (B2-L1 head, AlexNet, fine-tuned from the official checkpoint on aerial vehicles + pedestrians; seeds 20261023/20261102) as the final model of stages 2-6. Measured on the stage-2 sealed predictions (stored outputs, not reopened; gated by exact re-scoring):
- VisDrone-SOT test-dev, 34 sequences source-video-disjoint from training: success AUC **B0 0.527 → REF 0.602 (+0.075, 95% source-video cluster interval [+0.024, +0.124])**, both seeds positive.
- UAV123@10fps domain subset (76 sequences): B0 0.594, REF 0.589 (−0.005, [−0.034, +0.027]): no difference detectable.
- Eight further training arms and a 37-point post-processing grid were tested against REF on validation; none was adopted. The source-resolution hypothesis (R1) was not tested. These results are not comparable with published leaderboards.

## Wave 6 guarded execution — not approval ready

[Wave6 executable safeguards](<reports/wave6_progress.md>) add native Windows process-tree/inclusive-deadline supervision, exact one-use human acknowledgment and GPU lease, immediate runtime/input guards, and durable checkpoint/telemetry around the byte-unchanged scientific controller. [Compact evidence](<experiments/wave6_validation.json>) distinguishes CPU/math replay and native control tests from authentic training. **G5 remains blocked** by ambiguous WDDM C+G ownership; resource availability is checked, not assumed. No real SGD, GPU experiment, real grant or new approval-ready pilot proposal was issued. The new [guarded entry](<scripts/run_b2_guarded.py>) is the only supported future Wave6 production route; the mechanically retained legacy CLI is not an approved bypass. Establish actual prerequisites, freeze a fresh exact unapproved proposal, then wait for human approval.

Native follow-up: continuous claim/direct-parent lease checks now run at every safe boundary; strict inventory/resource validation closes reproduced malformed-output gaps. The [native review](<reports/wave6_native_review.md>) and [current evidence](<experiments/wave6_native_validation.json>) record **741 passing tests**, unchanged frozen science/assets, and real read-only probes. The physical adapter reports **WDDM 2.7**: its process-adapter `ClientHint` bytes are reserved, not a compute/graphics discriminator. Supported per-PID SM activity blends graphics and compute and cannot exclude idle contexts. This is an observed interface limitation—not proof of competing training, an allocation dispute, or a request for more attestations. G5 remains unresolved; no real SGD was run.

## Wave 5 preparation — historical execution block

[Wave5 preparation audit](<reports/wave5_progress.md>) preserves the frozen Wave4 contract and verifies439existing+18new tests (**457PASS**). No real-data SGD, pilot output or human grant exists. Independent audits identify mandatory cumulative-deadline, one-use authorization/GPU-lease, immediate-stop/invariant and durable-telemetry gaps in the old launcher. The exact hash-bound draft is **unapproved and not eligible for execution approval**; a reviewed additive safeguard and a newly frozen exact human grant are prerequisites. R1 remains on hold.

## Wave 4 decision (historical recommendation retained)

- Real1024candidate preflight: **973 genuine cross-frame positives,50 safe negatives,1 explicit crop-co-presence rejection,0 replacements**. All1024records/1023accepted archives independently replayed; source/target algebra independently audited.
- Separate fixed128attempt DEV control:64positive+64negative; negative draws contain38unique pairs and26repeats, not128independent examples. No frozen15sequence evaluation retuning or reruns.
- Actual Windows serial/spawn2 loading, image faults and consumed-cursor recovery pass. An authentic noncontiguous-input CUDA failure was preserved and repaired with the same pixel-preserving NCHW bridge used by the proposed pilot; bounded random-only synthetic CUDA replay passes. No asynchronous-prefetch or physical-batch-equivalence claim.
- Actual32TRAIN+128DEV forward/loss checks pass with unchanged parameters and **zero real backward/SGD**. All439 integrated tests pass;140priorfiles and26priorartifact references remain unchanged.
- **GO recommendation only** for a future separately human-authorized **100update/1600attempt-slot/900second** grader-only reliability pilot. [Frozen pilot](<configs/B2-Pilot-W4.json>) and [70epoch reference](<configs/B2-Reference-W4.json>) are distinct; the full reference remains unvalidated. No authorization file exists and the launcher still refuses real SGD without a new exact human grant.
- Measured full-input central budget3.20minutes, engineering range2.40–9.55minutes, hard15minutes; not convergence or tracking accuracy. Commands, signal/stop criteria and resume limitations are in the [handoff report](<reports/wave4_progress.md>).
- Raw data/models/arrays/run snapshots remain local ignored assets; a clean Git commit is not their backup. Hashes bind actual working bytes, including text EOL; preserve the recorded byte-identical source/config snapshots when transferring rather than silently accepting checkout newline conversions.

## Historical Wave 3 decision

- 1024 authentic training candidates: **973 accepted positives;51 quarantined negatives;0 replacements**. Zero accepted negatives and still-image-only data make the real training recipe unsafe.
- Twelve additional complete sequences,2672frames each: AUC **B0 0.519714 / B1 0.475797**. All adverse outcomes retained; no B1 promotion.
- Restricted synthetic CPU lifecycle replay is bit-exact; no production-worker/CUDA resume or physical-batch-equivalence claim.
- G1 bounded PASS; G2 PARTIAL; G3 PASS; G4 PARTIAL; G5 PARTIAL/not ready. H1 supported only as a limiting condition; H2 plausible but unconfirmed; H3 insufficient evidence.
- **Next: continue data/recipe correctness.** No real SGD, B2 pilot, R1 architecture or baseline retuning authorized. [Proposed recipe](<configs/B2-wave3-proposed.json>) remains launch-disabled.
- Integrated suite: **362 passed**. Raw data/models/run artifacts remain local ignored assets; committed compact evidence is not their backup.

## Established scope (Wave 2 evidence retained)

- Immutable official upstream submodule, commit `992aa7c1605360d607af84cb804b93cd8ee81caf`.
- Deterministic tensor, negative-target, score, crop geometry and evaluator correctness tests.
- Original CUDA inference runner and separately selectable **B1-score** sigmoid interpretation correction.
- Explicit B2 training correction patches, not applied to the official source or released checkpoint.
- Dataset import/hashing, immutable run provenance, sequence-level evaluator, and primary-source literature audit.
- Authoritative, bounded HTTP-range acquisition of three complete UAV123@10fps sequences (265 original decoded frames), with raw annotations and hashes preserved.
- Matched B0/B1-score pilot: success AUC **0.552222 / 0.551336**; precision@20px **0.693694 / 0.693694**. No aggregate improvement claim.
- Actual corrected B2 graph profiled at batches 1/2/4 with forward/backward/SGD; no OOM. Synthetic feasibility is not real training throughput or convergence.
- M1 and full M2 remain partial. Full B2/R1 training remains disabled; no R1 architecture was implemented.

See [initial progress](reports/initial_progress.md), [source audit](reports/source_audit.md), [inference/geometry audit](reports/inference_geometry_audit.md), [evaluation audit](reports/evaluation_audit.md), and [literature review](reports/literature_review.md).

## Setup (tested target: Windows 10 x64 / Git Bash / Python 3.13)

```bash
git submodule update --init --recursive
bash scripts/checkout_upstream.sh
bash scripts/setup_env.sh
.venv/Scripts/python.exe -m pytest
```

The exact transitive environment is frozen in [Windows/Python 3.13 lock](requirements/lock-windows-py313-cu128.txt). Setup disables pip cache after a traced Windows cache temporary-file creation loop; TLS verification remains enabled. Uses isolated PyTorch `2.10.0+cu128`; CUDA 12.8 wheels support this RTX 5070 Ti (Blackwell). Driver-reported CUDA maximum is not the installed PyTorch CUDA runtime. No global Python packages are modified. The first install downloads about 2.9 GB for torch; reserve >=10 GB temporary/install space. For Linux, adapt the interpreter path; Linux has not been tested. Original 2021 torch/CUDA numerical equivalence is **not** asserted. See [reproduction instructions](docs/reproduction.md).

CPU correctness-only alternative with existing Python:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

OpenCV/Pillow are included in the data/test extras; torch-dependent tests skip if torch is missing. For all439 tests use the pinned base environment plus torch and the preserved local authentic assets for optional real-data regression cases. Original B0 hardcodes CUDA and has not been silently replaced by a CPU baseline. The separate corrected B2 graph/lifecycle has CPU tests; these do not train on real data.

## Checkpoint and GPU smoke (not a benchmark)

```bash
.venv/Scripts/python.exe scripts/fetch_official_checkpoint.py
.venv/Scripts/python.exe scripts/diagnose_environment.py --output artifacts/environment.json
.venv/Scripts/python.exe scripts/smoke_inference.py --checkpoint artifacts/checkpoints/general_model.pth --output artifacts/runs/unique-smoke-id
```

Downloader is bounded, does not overwrite existing weights, and records the exact source URL and hash. If already downloaded, reuse the recorded file rather than rerunning it. Strict tensor-only checkpoint loading is required. Synthetic images test execution only, never accuracy/FPS claims.

## Real inference

Import actual benchmark frames/annotations using [dataset instructions](docs/datasets.md); declare coordinate origin and exact frame protocol. Export a GT-free sequence manifest with an ordered frame list and initial tracker-ready box. Then:

```bash
.venv/Scripts/python.exe scripts/run_sequence.py --sequence artifacts/imported-sequence.json --config configs/B0.json --output artifacts/runs/unique-B0-sequence-id
.venv/Scripts/python.exe scripts/run_sequence.py --sequence artifacts/imported-sequence.json --config configs/B1-score.json --output artifacts/runs/unique-B1-sequence-id
.venv/Scripts/python.exe scripts/evaluate.py --help
```

For the acquired pilot and its unresolved annotation-origin semantics, [Wave 2 data instructions](docs/wave2_data.md) and the explicit [native-numeric exporter](scripts/prepare_native_inputs.py) preserve literal upstream initialization/evaluation instead of silently subtracting one. [Matched driver](scripts/run_matched_subset.py), [CPU evidence audit](scripts/audit_wave2.py), and [bounded training profiler](scripts/profile_training.py) keep the experiment identities separate.

Only initial GT is permitted in inference manifests. Saved predictions include initialization; do not replace the first row with exact GT. CLI refuses existing run directories and invalid predictions, records source/config/checkpoint/frame hashes, synchronized inference and I/O-inclusive latency, and peak allocated/reserved GPU memory. Use the same checkpoint/config/sequences for the B0/B1 pair. See evaluator help for manifest and prediction paths.

## Baselines and scientific gates

| ID | Meaning | Enabled now |
|---|---|---|
| B0 | Released behavior + official weights | Original GPU execution; benchmark assets required |
| B1-score | B0 + sigmoid on BCE head before mixing | Inference-only ablation; no claimed gain |
| B2 | Validated training corrections + matched retraining | Bounded reliability-pilot recommendation; separate human SGD grant required; full retraining pending |
| R1 | Source-resolution-aware fusion on frozen B2 | Hypothesis/diagnosis only |

[Methodology](docs/methodology.md) defines matched controls, leakage safeguards, experiment records, and go/no-go gates. [Registry](experiments/registry.json) tracks status. Prior work already covers generic hierarchical fusion, template updates, crop adaptation and distillation; these are not standalone novelty claims.

## Layout

`src/hift_research/`: adapters, provenance, geometry, evaluation, opt-in corrections.
`configs/`, `experiments/`: separate identities and machine-readable records.
`scripts/`, `tests/`: reproducible commands and deterministic checks.
`reports/`, `docs/`: evidence and methodology.
`third_party/HiFT/`: immutable original (its own license applies).
`artifacts/`, `data/`: ignored local assets/results. Do not commit datasets or model weights.

No full training has been launched. See [AGENTS.md](AGENTS.md) for future-agent project rules.
