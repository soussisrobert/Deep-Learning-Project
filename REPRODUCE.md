# Reproducing the paper

Three levels. Level 1 is what a reader should run first; it needs nothing but Python.

| Level | Reproduces | Needs | Time |
|---|---|---|---|
| 1. Paper numbers | every mean, change and bootstrap interval in the paper | Python 3.10+, numpy, pandas | ~2 min, any laptop |
| 2. Re-score | per-video success AUC from saved tracker boxes | the datasets (ground truth) + each track's evaluator | ~10–30 min per benchmark |
| 3. Re-run | tracking (and, for domain, training) from checkpoints | GPU (night, domain), x86 CPU (int8 timing), datasets, checkpoints | hours |

## Level 1

```bash
pip install -r environment/requirements-verify.txt
python scripts/verify_paper_numbers.py --write     # 129/129 paper numbers reproduced
python scripts/check_evidence_checksums.py         # domain evidence bundle integrity
```

`results/` was produced from each track's own outputs by the scripts in `scripts/convert/`:

| Script | From | To |
|---|---|---|
| `night_to_common.py` | `tracks/night/evidence/per_video_*.csv` (official pysot OPE) | `results/night/` |
| `cpu_to_common.py` | `tracks/cpu_int8/analysis/all-rounds/per_video_success.csv` | `results/cpu/per_video_UAV123.csv` |
| `domain_to_common.py` | `tracks/domain_ft/evidence/stage7/artifacts/b2-eval/STAGE2-SEALED-*.evaluation.json` | `results/domain/` |

Statistics (all in `shared/stats.py`): paired percentile bootstrap, 10,000 resamples.
Night: videos, seed 20261118 (UAVDark135 pooled-39 = test rows then validation rows).
CPU: videos, seed 0; TEST-40 two-level (videos and calibration draws), seed 1.
Domain: source-video clusters (22 VisDrone, 48 UAV123), seed 20261105, every member sequence kept.
CPU per-video scores in `per_video_success.csv` are rounded to 4 decimals; every CPU number still
matches the paper at its printed precision.

## Level 2 and 3, per track

Common prerequisites: the official HiFT checkpoint (`checkpoints/README.md`) and the datasets
(`data/README.md`). Every sequence list used is in `data/splits/`.

### Night (`tracks/night`)

Code changes on HiFT 992aa7c: `pysot/utils/enhance.py` (gamma lookup table on uint8 crops),
`pysot/tracker/hift_tracker_tmpl.py` (anchored template: 0.8 × frame-1 + 0.2 × mean of last 3),
`tools/test_tmpl.py`, `tools/eval_nat.py`, `toolkit/datasets/nat.py`. Run from `tracks/night/tools`
with the checkpoint at `snapshot/first.pth`:

```bash
pip install -r ../requirement.txt
C="--config ../experiments/config.yaml"
G="--enh_template gamma --enh_search gamma --gamma 0.6"
AG="--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 $G"
for DS in NAT2021 NAT2021L UAVDark135; do
  python test.py      --dataset $DS --snapshot snapshot/first.pth $C     --tag _none   # original
  python test.py      --dataset $DS --snapshot snapshot/first.pth $C $G  --tag _g06    # gamma
  python test_tmpl.py --dataset $DS --snapshot snapshot/first.pth $C $AG --tag _ag     # gamma + anchor
  python eval_nat.py  --dataset $DS                                                     # official OPE
done
```

The exact cluster jobs are in `tracks/night/jobs/` (final evaluation: `job_night_final.sh`; overlap screen:
`job_uavdark_leak.sh`). The speed run (E26, job 55514) used the timer inside `tools/test_tmpl.py`; its
job script is not in the branch, but its logs are in `evidence/sources/speed_*.log`. Analysis scripts: `tools/illumination/E*.py` and `tools/*.py`
(loss stretches: `tools/lost_diag.py`; overlap screen: `tools/leak_one.py`). Logbook: `tracks/night/docs/experiments_log.md`.
Evidence package as delivered: `tracks/night/evidence/` (protocol, claims, selection log, sources).

### CPU int8 (`tracks/cpu_int8`)

See `tracks/cpu_int8/README.md` for the full TC2 (SLURM) procedure. Summary:

```bash
cd tracks/cpu_int8
bash setup_local.sh                         # env, pinned HiFT clone, tools/patch_hift.py (device-only patch)
# put the official checkpoint at weights/general_model.pth (sha256 in checkpoint.sha256)
python scripts/run_track.py --help          # one controlled condition
python scripts/retime_matched.py --help     # matched fp32 vs int8bb-s1 timing (Table 2)
python scripts/layer_sensitivity.py --help  # per-layer int8 sensitivity
```

Every round's plan, job queue, analysis script and output is under `analysis/` and `plan/`;
`RESULTS.md` summarises all rounds. Raw per-frame outputs for level 2 are the round archives listed in
`checkpoints/README.md` (release assets).

### Domain fine-tuning (`tracks/domain_ft`)

Code: `src/hift_research/`, configs in `configs/`, tests in `tests/`. The frozen evidence bundle is
`evidence/stage7/` (its own `BUNDLE_README.md`): sealed per-frame predictions and per-sequence
evaluations (`artifacts/b2-eval/`), training and validation run manifests (`artifacts/tc2-runs/`,
`artifacts/tc2-validation/`), stage decisions (`artifacts/stage3-diag … stage7/`), dataset manifests
and `source_cluster_mapping.csv`.

```bash
cd tracks/domain_ft
pip install -e .                                   # see requirements/ for pinned sets
python -m pytest -q tests                          # unit tests
cd evidence/stage7 && python scripts/stage7_closeout.py --help   # re-derive the closeout from artifacts
```

Training settings (from the run manifests): 10 epochs × 1,000 updates, batch 32, head learning rate
log-linear 1e-3 → 1e-4, seeds 20261023 and 20261102, starting from the official checkpoint.

## Not reproducible from released files

- The REF-C recovery and wrong-object counts that appeared in early drafts: their event list and
  scripts were not retained, so they were removed from the paper.
- Absolute timing depends on hardware; Table 2 is one Xeon Gold 6230R server, the night speed one A40.
