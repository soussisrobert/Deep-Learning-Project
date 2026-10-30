# Round 4 runbook (for the agent operating TC2)

**Goal:** run the gated round-4 queues. **Only submit what `scripts/r4_gate.py` lists under NEXT.**

The gate script also scores everything and cuts losing branches. When NEXT is empty, round 4 is finished: package the results.

**Do not change:**
- code, thresholds, or video lists;
- the DEV recipe choice. **Never edit `plan/round4/int8_selected.json`.**

- **Why these runs:** [ROUND4_PLAN.md](ROUND4_PLAN.md).
- **Queues and estimates:** [plan/round4/SCHEDULE.md](plan/round4/SCHEDULE.md).
- **TC2 rules:** as in round 2/3.
  - Two jobs at once, one GPU in total, 6 h wall.
  - `sbatch` only from the login node; no Python on the login node.
  - No chained `sbatch`.

## 1. Sync and check (laptop, VPN on)

The excludes protect the existing results and environment on TC2.

```bash
rsync -av --exclude .venv --exclude third_party --exclude env --exclude data --exclude 'results*' \
  --exclude runs --exclude logs --exclude weights --exclude .git ./ <user>@10.96.189.12:~/hift-stress/
```

On TC2:
- `ls plan/round4/queue/` lists **12** queue files.
- `plan/round4/int8_selected.json` must **not** exist yet.
- These earlier runs must exist, because round 4 compares against them:
  - `results/uav123-sub40_fp32_clean_nodef`
  - `results/uav123-sub40_fp32_pgd4s10_nodef`
  - `results/uav123-sub40_fp32_pgd4s50_nodef`
  - `results/uav123_fp32_clean_nodef_t4`

Then submit the GPU smoke check:

```bash
sbatch --partition=MGPU-TC2 --qos=normal --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:30:00 \
  --output=runs/r4_smoke_%j.out --wrap "source slurm/env_tc2.sh && python scripts/smoke_test.py --snapshot weights/general_model.pth"
```

Expect 17/17 (16/16 on a CPU-only machine). Stop if anything fails.

## 2. The loop

1. **Run the gate:**

   ```bash
   sbatch --partition=MGPU-TC2 --qos=normal --cpus-per-task=1 --mem=2G --time=00:15:00 \
     --output=runs/r4_gate_%j.out --wrap "source slurm/env_tc2.sh && python scripts/r4_gate.py"
   cat results/round4/criteria.md
   ```

2. **Submit queues from its NEXT list:**
   - `*_gpu.tsv` and `*_mix.tsv`: `sbatch --export=ALL,QUEUE=<file> slurm/job_queue_gpu_tc2.sbatch`
   - `*_cpu.tsv`: `sbatch --export=ALL,QUEUE=<file> slurm/job_queue_cpu_tc2.sbatch`
   - At most one GPU/mix job at a time, plus at most one CPU job beside it.
   - **First wave:** `J1_gpu` (about 10 min), then `J2_gpu` if J1 passes. In the CPU slot: `Q0b_cpu` (minutes), then `Q0_cpu` (about 4.4 h), then `J1c_cpu`.
3. **After every job:**
   - Check that the state is `COMPLETED` and the log ends `queue … finished, 0 failed`.
   - Check `sequences` is 40 for TEST runs and 20 for DEV runs.
   - Run the gate again.
4. **When a branch shows `CUT`,** never submit its queues, even if a slot is free.

**How the `int8sel` runs get their recipe:** they read it from `int8_selected.json`. The gate writes that file once the Q0 pilot and the Q0b benchmark are complete. If a Q1/C queue is ever started before that, its runs fail on purpose.

## 3. Report after every gate run

- Job IDs, states and elapsed times.
- `results/round4/criteria.md`.
- `plan/round4/int8_selected.json`, once it exists.

At the end, send a tarball of:
- every round-4 result folder;
- `results/edge_round4`, `results/round4` and `plan/round4`;
- the logs.

This is the same packaging as rounds 2–3.

## 4. Stop and ask if

- A run fails twice, or there is a Python traceback from `hiftstress/` or `scripts/`.
- A count is wrong: TEST must be 40 videos / 37,681 frames, DEV 20 videos / 16,826 frames.
- A job takes more than 1.5× its `SCHEDULE.md` estimate. The int8bb-s1 CPU rate is an estimate.
- Anything would need a different threshold, quality, recipe, ε, number of steps or video list.
