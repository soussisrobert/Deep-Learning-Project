# HiFT: trustworthy reproduction before improvement

Research implementation for UAV single-object tracking. The source-resolution-aware fusion idea is a **hypothesis**, not an established contribution. No real tracking benchmark or training result has been produced during initialization.

## Current scope

- Immutable official upstream submodule, commit `992aa7c1605360d607af84cb804b93cd8ee81caf`.
- Deterministic tensor, negative-target, score, crop geometry and evaluator correctness tests.
- Original CUDA inference runner and separately selectable **B1-score** sigmoid interpretation correction.
- Explicit B2 training correction patches, not applied to the official source or released checkpoint.
- Dataset import/hashing, immutable run provenance, sequence-level evaluator, and primary-source literature audit.
- B2/R1 training intentionally disabled pending validation. Missing benchmark data does not imply M2 success.

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

Optional torch/OpenCV-dependent tests explicitly skip if missing. Original baseline model hardcodes CUDA; a CPU model port has not been silently substituted.

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

Only initial GT is permitted in inference manifests. Saved predictions include initialization; do not replace the first row with exact GT. CLI refuses existing run directories and invalid predictions, records source/config/checkpoint/frame hashes, synchronized inference and I/O-inclusive latency, and peak allocated/reserved GPU memory. Use the same checkpoint/config/sequences for the B0/B1 pair. See evaluator help for manifest and prediction paths.

## Baselines and scientific gates

| ID | Meaning | Enabled now |
|---|---|---|
| B0 | Released behavior + official weights | Original GPU execution; benchmark assets required |
| B1-score | B0 + sigmoid on BCE head before mixing | Inference-only ablation; no claimed gain |
| B2 | Validated training corrections + matched retraining | Patches/tests only; no training launcher |
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
