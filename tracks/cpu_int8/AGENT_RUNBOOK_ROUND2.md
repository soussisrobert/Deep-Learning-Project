# Round 2 runbook (for the agent operating TC2)

**Goal:** run the round-2 queues on TC2, check each result, and report the verdicts from `scripts/evaluate_round2.py`.
All code is ready and tested. **Do not change code or thresholds.** If something breaks, stop and report it (see section 5).

- **Why these runs:** [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
- **Per-queue contents and time estimates:** [plan/SCHEDULE.md](plan/SCHEDULE.md).

## 0. Rules (TC2)

**Submitting jobs**
- Submit with `sbatch` from the login node, in `~/hift-stress`.
- Never run Python, compute or `nvidia-smi` on the login node. `squeue`, `sacct`, `cat`, `tail` and `ls` are fine.
- Never call `sbatch` inside a job.
- No `sudo`. Use only the private env at `~/hift-stress/env`.

**Limits**
- At most **2 jobs at once**, with at most 1 GPU, 10 CPUs and 30 GB between them.
- Pair **one GPU queue job** (6 CPU / 20 GB) with **one CPU queue job** (4 CPU / 8 GB).
- Each job has a 6 h wall.

**Data**
- Never print or copy any Hugging Face token.
- Never commit `weights/`, `data/`, `results/` or `runs/` to git.

## 1. Sync the code (on the laptop, VPN on)

The repo gained new scripts in round 2, so TC2 needs this sync. The excludes protect the round-1 results already on TC2.

```bash
rsync -av --exclude .venv --exclude third_party --exclude env --exclude data --exclude 'results*' \
  --exclude runs --exclude logs --exclude weights --exclude .git ./ <user>@10.96.189.12:~/hift-stress/
```

Then check these on TC2:
- `ls scripts/run_queue.sh scripts/evaluate_round2.py plan/queue/` shows the files, with **10 queue files**.
- `ls results/` still lists the 11 round-1 runs.

Round 1 used these baseline run names. `evaluate_round2.py` needs them, so stop if any is missing:

| Run name | Used as |
|---|---|
| `uav123-sub40_fp32_clean_nodef` | Clean baseline |
| `uav123-sub40_fp32_pgd4s10_nodef` | PGD baseline |
| `uav123_fp32_clean_nodef_t4` | fp32 CPU reference |

## 2. Submit, in this order

Each step starts only after the job before it in the same slot has finished.

| Step | Slot A (GPU) | Slot B (CPU) |
|---|---|---|
| 1 | `sbatch --export=ALL,QUEUE=plan/queue/01_gpu_P1.tsv slurm/job_queue_gpu_tc2.sbatch` | `sbatch slurm/job_rescore_tc2.sbatch` (about 1.5 h; also the first test of the new code) |
| 2 | `02_gpu_P1.tsv` | `sbatch --export=ALL,QUEUE=plan/queue/05_cpu_P2.tsv slurm/job_queue_cpu_tc2.sbatch` |
| 3 | `03_gpu_P1.tsv` | `10_cpu_P5.tsv` (CPU job) |
| 4 | `04_gpu_P1.tsv` | (free) |
| 5 | `06`, `07`, `08` (P3) | (free) |
| 6 | `09_mix_P4.tsv`, using the **GPU** job script (its runs use GPU + 4 CPU threads) | (free) |

- **If the deadline is close:** steps 1–4 plus queue 05 answer both main questions. P3, P4 and P5 strengthen them.
- **Resuming:** a queue skips runs that already have `results/<run>/summary.json`. If a job hits the 6 h wall, re-submit the **same queue**. A run killed mid-way restarts from scratch.

## 3. Check every finished job (read the log before submitting the next one)

| Check | Command or source | Good if |
|---|---|---|
| Job state | `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS` | `COMPLETED` |
| Last log line | `tail -5 runs/queue_gpu_<id>.out` (or `queue_cpu_`, `rescore_`) | `queue ... finished, 0 failed` |
| Each run's sequence count | `results/<run>/summary.json` → `sequences` | 40 (`uav123-sub40_*`) or 123 (`uav123_*`) |
| Clean runs, sanity | `success_auc` of `fp32_clean_<defence>` | Between 0.45 and 0.60 (clean baseline 0.582). Far outside → report it |
| Job 0 | `results/rescore/` and `results/edge_round2/cpu_t4_int8full.json` | Rescore says the stored numbers reproduce. Official precision on the full fp32 CPU run is near HiFT's 0.787 |
| Time vs estimate | `Elapsed` vs `plan/SCHEDULE.md` | Within about 30%. If much slower, report before continuing |

After each step, submit a CPU job that runs the scorer. It takes seconds, but the login node is not for Python:

```bash
sbatch --partition=MGPU-TC2 --qos=normal --cpus-per-task=1 --mem=2G --time=00:15:00 \
  --output=runs/eval_%j.out --wrap "source slurm/env_tc2.sh && python scripts/evaluate_round2.py"
cat results/round2/criteria.md
```

## 4. What to send back after each step

- **Job IDs**, states, elapsed times, and any `FAILED` run names with the last 30 lines of the log.
- **Scorer output:** the full `results/round2/criteria.md`.
- **Round-2 files:** a tarball of the new results, without the large box files. Example for the P1 runs:

  ```bash
  tar czf round2_results.tgz results/round2 results/rescore results/edge_round2 \
    $(ls -d results/uav123-sub40_fp32_*_{blur1,median3,bits5,jpeg75}* 2>/dev/null)
  ```

  Adjust the pattern for other queues.

**Report only what the files show.** Mark any run that is incomplete or failed as such. Do not estimate missing numbers.

## 5. Stop and ask (do not fix it yourself) if

- A job fails twice on the same run.
- `sequences` count is wrong, or the sanity check in section 3 fails.
- The rescore check says the stored numbers do not reproduce.
- A Python error comes from `hiftstress/` or `scripts/`. Report the traceback; the code owner fixes it.
- Something would need a different threshold, sequence list, ε or step count. These are pre-registered on purpose.
