# Round 3 runbook (for the agent operating TC2)

**Goal:** run the round-3 queues, make the pre-registered JPEG-quality choice between the two stages, check every result, and report the verdicts from `scripts/evaluate_round3.py`.

**Do not change code, thresholds, the DEV/TEST video lists or the selection rule.** If something breaks, stop and report it (section 6).

- **Why these runs:** [ROUND3_PLAN.md](ROUND3_PLAN.md).
- **Queue contents and time estimates:** [plan/round3/SCHEDULE.md](plan/round3/SCHEDULE.md).
- **TC2 rules:** the same as round 2 ([AGENT_RUNBOOK_ROUND2.md](AGENT_RUNBOOK_ROUND2.md), section 0).
  - At most 2 jobs at once, 1 GPU in total, 6 h per job.
  - `sbatch` only from the login node; no Python on the login node.
  - Never chain `sbatch` inside a job.

## 1. Sync the code (laptop, VPN on)

The excludes protect the existing results and environment on TC2.

```bash
rsync -av --exclude .venv --exclude third_party --exclude env --exclude data --exclude 'results*' \
  --exclude runs --exclude logs --exclude weights --exclude .git ./ <user>@10.96.189.12:~/hift-stress/
```

On TC2, check that the sync worked:
- `ls plan/round3/queue/` lists **17** queue files.
- `ls scripts/r3_select.py scripts/evaluate_round3.py hiftstress/torchjpeg.py hiftstress/corruptions.py` shows all four files.

Check that these round-2 runs exist, because round 3 compares against them. **Stop if any is missing.**
- `results/uav123-sub40_fp32_clean_nodef`
- `results/uav123-sub40_fp32_pgd4s10_nodef`
- `results/uav123-sub40_fp32_clean_jpeg75`
- `results/uav123-sub40_fp32_pgd4s10_jpeg75_adaptive`
- `results/uav123-sub40_int8bb_clean_nodef_t4`
- `results/uav123-sub40_int8bb_pgd4s10_nodef_t4`
- `results/uav123_fp32_clean_nodef_t4`
- `results/edge_round2/cpu_t4_fp32.json` and `cpu_t4_int8bb.json`

First job, a quick GPU check of the new code (about 10 min):

```bash
sbatch --partition=MGPU-TC2 --qos=normal --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:30:00 \
  --output=runs/r3_smoke_%j.out --wrap "source slurm/env_tc2.sh && python scripts/smoke_test.py --snapshot weights/general_model.pth"
```

Expect **16/16** on the GPU (15/15 on a CPU-only machine). **Stop if anything fails.**

## 2. Stage 1

Submit with these two commands, swapping in the queue file:

```bash
sbatch --export=ALL,QUEUE=<file> slurm/job_queue_gpu_tc2.sbatch   # for gpu queues
sbatch --export=ALL,QUEUE=<file> slurm/job_queue_cpu_tc2.sbatch   # for cpu queues
```

Order (one GPU job at a time; the second slot takes the CPU job, or waits):

| Step | GPU slot | Other slot |
|---|---|---|
| 1 | `01_S1_gpu_P1.tsv`: DEV sweep (needed for the selection) | `02_S1_cpu_P1.tsv`: CPU timing |
| 2 | `05_S1_gpu_P2.tsv` | – |
| 3 | `03_S1_gpu_P2.tsv`, then `04_S1_gpu_P2.tsv` (PGD-50 undefended, 2 parts) | – |
| 4 | `06_S1_gpu_P2.tsv`, then `07_S1_gpu_P2.tsv` (JPEG75 vs surrogate-50, 2 parts) | – |
| optional | `08_S1_gpu_P7.tsv`, `09_S1_gpu_P7.tsv` (BPDA-50 on JPEG75) | – |

Split runs (`__p1`, `__p2`) merge automatically at the end of each queue job.

## 3. Selection (after stage-1 jobs 1–7 have finished)

```bash
sbatch --partition=MGPU-TC2 --qos=normal --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output=runs/r3_select_%j.out --wrap "source slurm/env_tc2.sh && python scripts/r3_select.py"
cat runs/r3_select_<id>.out plan/round3/selected.json
```

- The script uses DEV videos only and the rule fixed in `ROUND3_PLAN.md`. **Never edit `selected.json` by hand.**
- If it picks 75, it links the stage-2 runs that are identical to existing JPEG-75 runs, so they are not run twice.

## 4. Stage 2 (only after `selected.json` exists)

| Step | GPU slot | Other slot |
|---|---|---|
| 5 | `10_S2_gpu_P3.tsv` | `15_S2_cpu_P4.tsv`: int8 calibration sets and int8 + JPEG clean |
| 6 | `11_S2_gpu_P3.tsv`, then `12_S2_gpu_P3.tsv`, then `13_S2_gpu_P3.tsv` | – |
| 7 | `14_S2_mix_P4.tsv`: GPU job script; the int8 model runs on the CPU, the attack on the GPU | – |
| 8 | `16_S2_gpu_P5.tsv`: corruptions | – |
| 9 | `17_S2_gpu_P6.tsv`: clean cost of the other qualities, plus seeds | – |

- **Before any stage-2 job:** `selected.json` must exist. If it is missing, the stage-2 commands fail on purpose with an argparse error.
- **Resuming:** re-submitting a queue skips runs that are already finished, and parts whose merged run exists.

## 5. After every job

Check each job the same way as round 2 (runbook section 3):
- The job state is `COMPLETED`.
- The last log line reads `queue … finished, 0 failed`.
- `sequences` in `summary.json` is 40 for TEST runs and 20 for DEV runs.

Then score the results so far:

```bash
sbatch --partition=MGPU-TC2 --qos=normal --cpus-per-task=1 --mem=2G --time=00:15:00 \
  --output=runs/r3_eval_%j.out --wrap "source slurm/env_tc2.sh && python scripts/evaluate_round3.py"
cat results/round3/criteria.md
```

**Send back after each step:**
- Job IDs, states and elapsed times.
- Any `FAILED` runs, with the last 30 lines of their logs.
- `results/round3/criteria.md`.
- `plan/round3/selected.json`, once it exists.

**At the end:** a tarball of every round-3 result folder plus `results/bench`, `results/round3` and the logs. This is the same packaging as the complete round-2 handoff.

## 6. Stop and ask (do not fix it yourself) if

- **A run fails twice**, or a Python error comes from `hiftstress/` or `scripts/`. Send the traceback.
- **A video count is wrong**: TEST must have 40 videos and 37,681 frames, DEV 20 videos and 16,826 frames.
- **`r3_select.py` reports a missing DEV run**, or anything suggests TEST data influenced the selection.
- **A queue would need a different ε, number of steps, quality list or threshold.**
- **A job takes more than 1.5× its estimate in `SCHEDULE.md`.** Report it before continuing; the 50-step and surrogate costs are estimates.
