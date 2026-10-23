# HiFT: trustworthy reproduction before improvement

Research implementation for UAV single-object tracking. Source-resolution-aware fusion remains a **hypothesis**, not an established contribution. **Wave 3** adds a canonical real-input preflight, synthetic lifecycle/resume verification and twelve additional frozen B0/B1 sequences—not real-data training or full benchmark reproduction. See [Wave 3 delivery and gates](<reports/wave3_progress.md>), [compact evidence](<experiments/wave3_validation.json>) and unchanged [Wave 2 results](<reports/wave2_progress.md>).

## Wave 3 decision

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

OpenCV/Pillow are included in the data/test extras; torch-dependent tests skip if torch is missing. For all362 tests use the pinned base environment plus torch. Original B0 hardcodes CUDA and has not been silently replaced by a CPU baseline. The separate corrected B2 graph/lifecycle has CPU tests; these do not train on real data.

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
| B2 | Validated training corrections + matched retraining | Isolated synthetic feasibility only; real-data recipe/retraining pending |
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
