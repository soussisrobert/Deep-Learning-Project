# CPU efficiency track: int8bb-s1 (static int8 backbone, layer1 and head fp32)

**Paper section:** CPU efficiency; Table 2; "Why HiFT fails" (confidence warning, PGD attack).
**Base:** HiFT commit `992aa7c` (cloned and patched by `setup_local.sh` / `tools/patch_hift.py`;
4 device-only lines in 2 files, `analysis/2026-11-02_hift-pin-check/PIN_CHECK.md`), official checkpoint.

## What changes
PyTorch FX static post-training quantisation (fbgemm) of the AlexNet backbone, calibrated on
40 non-evaluation videos × 4 spread frames; `layer1` and the complete head stay fp32
(`hiftstress/edge.py`). Nothing else in HiFT changes.

## Selection vs report
DEV-20 chose the recipe (round 3); TEST-40 reports it (round 4, three calibration draws); all-123
pools TEST with a separately calibrated non-TEST model (round 5). Lists: `../../data/splits/cpu/`.

## Run
`README.md` (TC2/SLURM procedure), `REPRODUCE.md` at the repository root (summary).
Matched timing: `scripts/retime_matched.py`, `slurm/job_retime_tc2.sbatch`.
Per-layer sensitivity: `scripts/layer_sensitivity.py`.

## Evidence
- `RESULTS.md`: all rounds on one page.
- `analysis/<date>_<round>/`: each round's REPORT.md, analysis script, its printed output and numbers.
- `analysis/2026-11-02_retime-matched/`: the timing in Table 2.
- Per-video results in the common format: `../../results/cpu/`.
