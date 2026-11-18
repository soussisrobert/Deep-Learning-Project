# HiFT-stress: verified TC2 handoff

> **New here? Read [TEAM_HANDOFF.md](TEAM_HANDOFF.md) first.**
>
> **Just want the results? See [RESULTS.md](RESULTS.md)** (all five rounds on one page).

## What worked

Verified on TC2 with NVIDIA A40, Xeon Gold 6230R, Python 3.11.17 and torch 2.8.0+cu128. HiFT is pinned to `992aa7c1605360d607af84cb804b93cd8ee81caf`; `tools/patch_hift.py` applies the device changes needed for inference. The GPU smoke test passed **14/14** checks.

Use this updated package rather than the original ZIP. It includes the converter, optional streaming data download, strict dataset counts, exact condition selection through `RUN_ONLY`, and separate CPU jobs.

## Replicate on your own TC2 account

1. Unpack this ZIP so the project is exactly `~/hift-stress`. Supply the official HiFT checkpoint as `weights/general_model.pth`. Our run used an existing `first.pth`, copied to that name. Its SHA-256 is in `checkpoint.sha256`; the automatic weight download was **not verified**. Data, weights, the environment and third-party HiFT checkout are excluded from this ZIP.
2. From `~/hift-stress`, run these **one at a time**, checking each job's exit status and log before continuing:

```bash
mkdir -p runs logs weights
sbatch slurm/job_setup_tc2.sbatch
sbatch --export=ALL,STREAM_DOWNLOAD=1 slurm/job_data_uav123_tc2.sbatch
sbatch slurm/job_smoke_tc2.sbatch
```

Setup creates `~/hift-stress/env`, installs dependencies, clones the pinned HiFT code and patches it. The data job downloads UAV123 and converts it. Expect **123 sequences / 112578 frames**. Streaming avoids storing an additional 13 GiB archive; interrupted streaming must restart. `requirements.txt` is the original top-level dependency list; `environment-tested.txt` records the versions actually installed, for your agent to pin or compare. This ZIP has not been tested through a fresh installation.

3. The tested 40-sequence GPU batch below finished in **4h01m**. Selection is spread across the dataset, not the first 40 videos; see `attack-subset-40.json`.

```bash
RUN_ONLY='fp32_clean_nodef fp32_noise2_nodef fp32_noise4_nodef fp32_noise8_nodef fp32_fgsm1_nodef fp32_fgsm2_nodef fp32_fgsm4_nodef fp32_fgsm8_nodef fp32_pgd4s10_nodef' \
sbatch --cpus-per-task=6 --mem=20G --export=ALL,PHASE=A,MAXSEQ=40,MAXFRAMES=,OUT=results slurm/job_adv_tc2.sbatch
```

The tested CPU batch finished in **3h16m**. It measures four variants at 1/2/4 threads, then runs full-dataset CPU fp32 clean tracking. It does not run a CPU attack.

```bash
sbatch --export=ALL,CPU_STAGE=bench_clean,CPU_VARIANT=fp32,OUT=results slurm/job_cpu_edge_tc2.sbatch
```

## Limits and results

TC2's normal QoS permits **two submitted jobs total**, six hours per job, and aggregate 10 CPUs / 30 GB / one GPU. The two batches above fit together at 6+4 CPUs and 20+8 GB. Confirm limits on your own account. Submit from the login node; do not submit new jobs inside compute jobs. Check your account quota; shared filesystem free space is not your quota. Completed conditions are skipped on resubmission, but an interrupted condition restarts.

Full clean AUC: GPU **0.5838**, CPU **0.5865**. On the same 40 videos: clean **0.5825**, noise-4 **0.5625**, FGSM-4 **0.4177**, PGD-4 **0.0955**. PGD uses 10 steps; eps is pixel levels out of 255 on the search crop; seed 0. CPU int8-backbone network latency was **42.71 ms** versus **89.43 ms** fp32 at four threads. These are tracking quality and cost measurements, not flight tests. Completed tables are in `verified-results/`; do not mix 5-, 40- and 123-video comparisons.

## Leave these to the receiving agent

Remaining PGD budgets; defence/adaptive-attack grid; full GPU compression benchmarks/accuracy; full int8 tracking accuracy; and CPU attack splitting. CPU PGD measured 0.478 fps, projecting to about **22 hours** for 40 videos, so the existing monolithic CPU attack job will not fit. Mac/MPS portability and fresh installation of this ZIP are unverified. No retraining or automatic dispatcher was run.

For changes: `hiftstress/attacks.py`, `defenses.py`, `edge.py` and `tracker.py`; use `scripts/smoke_test.py`, then `run_track.py` for one controlled condition. Keep the checkpoint, sequence manifest and seed fixed when comparing changes.
