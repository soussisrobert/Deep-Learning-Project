# Round 5 runbook (for the agent operating TC2)

**Goal:** run the round-5 queues. **Only submit what `scripts/r5_gate.py` lists under NEXT.**

The gate script also scores everything and cuts losing branches. When NEXT is empty, round 5 is finished: package the results.

**Do not change:**
- code, thresholds, or video lists;
- `plan/round5/threshold.json` or `plan/round5/attacker_selected.json` (the gate writes them from DEV only).

- **Why these runs:** [ROUND5_PLAN.md](ROUND5_PLAN.md).
- **Queues and estimates:** [plan/round5/SCHEDULE.md](plan/round5/SCHEDULE.md).
- **TC2 rules:** as before.
  - Two jobs at once, one GPU in total, 6 h wall.
  - `sbatch` only from the login node; no Python on the login node.
  - No chained `sbatch`.

**Round 4 is closed.** `J1c_cpu.tsv` must never be run (it belongs to the CUT JPEG-85 branch). Its open questions are now handled by the R4C queues here.

## 1. Sync and check (laptop, VPN on)

The excludes protect the existing results and environment on TC2.

```bash
rsync -av --exclude .venv --exclude third_party --exclude env --exclude data --exclude 'results*' \
  --exclude runs --exclude logs --exclude weights --exclude .git ./ <user>@10.96.189.12:~/hift-stress/
```

On TC2:
- `ls plan/round5/queue/` lists **9** queue files.
- `plan/round5/threshold.json` and `plan/round5/attacker_selected.json` must **not** exist yet.
- These earlier runs must exist, because round 5 compares against them:
  - `results/uav123-sub40_fp32_clean_nodef`
  - `results/uav123_fp32_clean_nodef_t4`
  - `results/uav123-sub40_int8sel_clean_nodef_t4_cal1`, `_cal2` and `_cal3`

Then submit the GPU smoke check:

```bash
sbatch --partition=MGPU-TC2 --qos=normal --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:30:00 \
  --output=runs/r5_smoke_%j.out --wrap "source slurm/env_tc2.sh && python scripts/smoke_test.py --snapshot weights/general_model.pth"
```

Expect 18/18 (17/17 on a CPU-only machine). Stop if anything fails.

## 2. The loop

1. **Run the gate.** It needs the dataset for ground truth, so always run it through Slurm:

   ```bash
   sbatch --partition=MGPU-TC2 --qos=normal --cpus-per-task=1 --mem=4G --time=00:20:00 \
     --output=runs/r5_gate_%j.out --wrap "source slurm/env_tc2.sh && python scripts/r5_gate.py"
   cat results/round5/criteria.md
   ```

2. **Submit queues from its NEXT list:**
   - `*_gpu.tsv`: `sbatch --export=ALL,QUEUE=<file> slurm/job_queue_gpu_tc2.sbatch`
   - `*_cpu.tsv`: `sbatch --export=ALL,QUEUE=<file> slurm/job_queue_cpu_tc2.sbatch`
   - One GPU job at a time, plus at most one CPU job beside it.
   - **Suggested order:**
     - GPU: `P0` (about 1 h), then whatever the gate releases (`P1`, `F1`, `A1`, `A2`). `K1` (about 3.5 h) whenever the GPU would otherwise sit idle.
     - CPU: `R4C1` (minutes), then `R4C2` (about 3.7 h), then `K2` (about 4.4 h).
3. **After every job:**
   - Check that the state is `COMPLETED` and the log ends `queue … finished, 0 failed`.
   - Check `sequences`: 40 for TEST, 20 for DEV, 83 for `uav123-nt83_*`.
   - Run the gate again.
4. **When a branch shows `CUT`,** never submit its queues.

## 3. Report after every gate run

- Job IDs, states and elapsed times.
- `results/round5/criteria.md`.
- `plan/round5/*.json`, once they exist.

At the end, send a tarball of:
- every round-5 result folder, including `scores/`;
- `results/round5`, `results/bench` and `results/attributes_per_video.json`;
- `plan/round5`;
- the logs.

## 4. Stop and ask if

- A run fails twice, or there is a Python traceback.
- A count is wrong: TEST must be 40 videos / 37,681 frames, DEV 20 / 16,826, non-TEST 83 / 74,897.
- `export_attributes.py` finds no labels.
- A job takes more than 1.5× its estimate.
- Anything would need a different threshold, λ list, ε, number of steps or video list.
