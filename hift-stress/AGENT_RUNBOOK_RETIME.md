# Runbook: matched re-timing (CPU only, 2 jobs, about 3.5 h in total)

## Why
- The paper's speed claim (int8bb-s1 1.80× network, 1.76× tracker step) has two gaps:
  - The network timing used an int8 model that `bench_edge.py` calibrated itself (20 videos), not the accuracy-tested one (40 videos × 4 spread frames, set 1).
  - The fp32 tracker-step time came from a different job.
- `scripts/retime_matched.py` closes both gaps:
  - It rebuilds the accuracy-tested model through the same code path.
  - It **proves** that model is the one that was tested: the calibration videos must match those saved in the accuracy run, and its boxes are compared with that run's saved boxes.
  - It times fp32 and int8 in the same process, with the order alternating across repeated sessions.

**Do not change code, flags, video lists or thresholds.**

## Before submitting (TC2 login node)
1. Sync from the laptop as usual (same rsync excludes as round 5).
2. Check that these exist:
   - `scripts/retime_matched.py` and `slurm/job_retime_tc2.sbatch`;
   - `results/uav123-sub40_int8sel_clean_nodef_t4_cal1/summary.json` and its `preds/`;
   - `results/uav123_fp32_clean_nodef_t4/preds/`.
3. **Nothing else of ours may run at the same time** (no GPU job either). The timings must not compete for the CPU.

## Run (one job at a time)
```bash
sbatch --export=ALL,THREADS=4,SESSIONS=6 slurm/job_retime_tc2.sbatch
# when it has finished (COMPLETED), then:
sbatch --export=ALL,THREADS=1,SESSIONS=4 slurm/job_retime_tc2.sbatch
```

## Checks after each job
- The log must print `calibration videos identical: **True**`.
  - If it stops with "STOP: calibration videos differ", report it and do not retry with other settings.
- In the identity lines, int8 boxes should match the saved run (values near 1.000 on most videos).
  - A video can diverge if the node's CPU differs from the original run's. Report the values; don't act on them.
- `lscpu` in the log should show the same CPU model as the earlier CPU jobs (Xeon Gold 6230R). If it is different, report it.

## Send back
- `results/retime_matched_t4/` and `results/retime_matched_t1/`, each with `REPORT.md`, `summary.json` and `sessions.csv`.
- Both job logs.
- Job IDs, node names and elapsed times.

**Stop and ask** if a job fails, takes more than 1.5× its estimate, or the CPU model differs.
