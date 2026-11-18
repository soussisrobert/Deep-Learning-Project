# Reproduction and asset provenance

## Source identity

Official repository: https://github.com/vision4robotics/HiFT . Pinned source SHA: `992aa7c1605360d607af84cb804b93cd8ee81caf`. Parent repository stores a real git submodule, not an edited vendor copy. Run `bash scripts/checkout_upstream.sh` before measurements. All corrections remain outside the submodule; `git -C third_party/HiFT status --short` should show no tracked modifications.

The pinned README states Ubuntu 18.04 / Python 3.8.3 / torch 0.7.0 or 1.6.0 / CUDA 10.2; preserve that statement as upstream documentation, not an independently verified compatible combination. The dependency file is actually `requirement.txt`, despite README saying `requirements.txt`.

## Local compatibility scope

Test target is Windows 10 Enterprise LTSC 10.0.19044, Python 3.13.0, RTX 5070 Ti. Use `scripts/setup_env.sh` for an isolated CUDA 12.8 torch 2.10.0 environment. Original Python torch 1.x/CUDA 10.2 is not suitable for this Blackwell GPU. Current ports do not modify model equations; nevertheless kernel/attention numerics across torch versions are not yet compared, so original historical computation equivalence is unresolved.

External runner replaces dataset discovery and result saving, avoiding optional toolkit Cython overlap extensions not needed for OPE. It instantiates original ModelBuilder/HiFTTracker on GPU. No layout, crop, penalty, minimum box-size or raw-score change occurs in B0. Deserialization uses `weights_only=True`, CPU map_location and strict state_dict validation followed by device copy. The released checkpoint also contains optimizer metadata: static inspection found only `numpy.dtype` and the legacy `numpy.core.multiarray.scalar` beyond default globals. A context-local allowlist permits that scalar constructor (old/new import aliases), `numpy.dtype`, and `Float64DType`; unrestricted pickle loading is never used. NumPy 2.4 emits a legacy dtype alignment deprecation warning during this safe load; the warning is retained, not hidden. Prefix normalization only strips `module.`; key collisions fail. Any missing key/shape mismatch fails rather than silently initializing weights.

`B1-score` wraps model.track output and applies sigmoid to cls2 only. Same original checkpoint; original layout remains. No calibration/retuning or geometry correction is bundled.

## Official full checkpoint

Pinned README line 23 links Google Drive ID `1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA` (and Baidu mirror). `scripts/fetch_official_checkpoint.py` uses that exact Drive asset and records the acquisition time, content type, bytes, SHA-256. No author-published checksum was found; the local hash establishes identity, not independent authorship verification.

Acquired during initialization: **86,039,991 bytes**, SHA-256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`. Downloaded from `https://drive.usercontent.google.com/download?id=1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA&export=download&confirm=t` at 2026-10-23 10:32:39 UTC. Strict key load/GPU evidence is reported separately; download alone is not a baseline reproduction. Upstream includes `pretrained_models/alexnet-bn.pth`; that is NOT a substitute for general_model.

## Minimum data to unblock M2 pilot

One complete unmodified benchmark sequence (not cherry-picked successful frames): ordered images, original xywh annotations, exact dataset/version and sequence ID, frame range/stride, coordinate origin, annotation visibility/absence semantics, and dataset permission/source. For full benchmark numbers, import all official sequences under the correct protocol. UAV123@10fps must use official frame/annotation alignment; never apply an arbitrary stride to UAV123 and assume equivalence. UAV20L is long-term one-pass evaluation, not a VOT reset protocol.

The dataset importer is local-only and hashes assets; it does not silently download substitutes. Official dataset source and conversion details are in docs/datasets.md. No global disk scan beyond this project was performed; external user data locations are unknown.

## Commands and result identities

```bash
bash scripts/checkout_upstream.sh
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/smoke_inference.py --checkpoint artifacts/checkpoints/general_model.pth --output artifacts/runs/unique-official-smoke
.venv/Scripts/python.exe scripts/run_sequence.py --sequence artifacts/sequence-inference.json --config configs/B0.json --output artifacts/runs/unique-B0
.venv/Scripts/python.exe scripts/run_sequence.py --sequence artifacts/sequence-inference.json --config configs/B1-score.json --output artifacts/runs/unique-B1
.venv/Scripts/python.exe scripts/evaluate.py --help
```

Do not overwrite unique run directories. Benchmark results include all submitted prediction rows incl initialization. `initial_bbox` must already be the tracker-ready zero-based box from the declared upstream initialization convention, not guessed GT. Import/export documents the original half-pixel conversion.

## Training preflight (not an implemented recipe)

Pinned config uses batch 220 over two GPUs, 70 epochs, 600,000 samples/epoch, SGD/LR settings and four training sources. That is not feasible to copy blindly onto one 16GB GPU. Before M3: validate dataset crop generation, single-device dynamic-batch decode (currently static BATCH_SIZE/NUM_GPU), all corrections, optimizer groups/frozen layers/BN, resume RNG/optimizer/scheduler state, and matched original-code retrain. Start a single GPU batch1/2/4 memory/throughput sweep with backward passes. Inference memory/latency cannot establish training memory or duration. Estimate runtime as effective epochs*samples_per_epoch/measured_train_samples_per_second plus data/validation overhead; no numerical duration claim before measurement. Precropped datasets likely exceed the currently free project disk and need a storage plan before download.
