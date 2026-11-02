# Runbook: per-layer int8 sensitivity (CPU only, 3 Slurm jobs, about 4 h wall)

**Why:** `LAYER_SENSITIVITY_PLAN.md`. **Do not change code, flags, video lists or settings.**

## Before submitting
1. Sync from the laptop as usual (same rsync excludes).
2. **Wait until the re-timing job (1-thread, `hift_retime`) has finished.** Nothing may run beside it while it times.
3. Check that these exist:
   - `scripts/layer_sensitivity.py`, `slurm/job_layersens_tc2.sbatch`;
   - `results/uav123-dev20_int8bb-s1_c40x4s_cal{1,2,3}/` and `results/uav123-dev20_int8bb_c40x4s_cal{1,2,3}/` (round 4);
   - `results/uav123-dev20_fp32_clean_nodef_t4/`.

## Run
1. **Stage A alone** (its latency column needs a quiet CPU):
   ```bash
   sbatch --export=ALL,STAGE=fidelity slurm/job_layersens_tc2.sbatch
   ```
   - The log must show `draw<k>_calibration_matches_round4: True` for draws 1–3, and box-match values near 1.0.
   - If it prints `STOP: ... calibration differs`, report it; do not retry with other settings.
2. **After Stage A has COMPLETED,** submit both tracking jobs. They may run at the same time (accuracy is unaffected; their ms/frame is descriptive only):
   ```bash
   sbatch --export=ALL,STAGE=track,DRAWS="1" slurm/job_layersens_tc2.sbatch
   sbatch --export=ALL,STAGE=track,DRAWS="2 3" slurm/job_layersens_tc2.sbatch
   ```
   The tracking stage is resumable: if a job hits the time limit, resubmit the same command and finished runs are skipped.
3. **Final report:** each job rebuilds `results/layer_sensitivity/REPORT.md` at its end, so after the last job, that file is complete.

**Check:** each of `results/layer_sensitivity/track/keep{2,3,4,5}_cal{1,2,3}/` has `per_seq.json` with 20 videos.

## Send back
- `results/layer_sensitivity/` (`REPORT.md`, `fidelity.csv`, `fidelity_meta.json`, `track/*/summary.json` and `per_seq.json`; preds optional);
- all job logs;
- job IDs, node names, states and elapsed times.

**Stop and ask** if a job fails twice, there is a traceback, a job takes more than 1.5× its estimate, or an identity check fails.
