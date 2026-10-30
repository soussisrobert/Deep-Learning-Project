# hift-stress: stress-testing HiFT for drone ISR

> **New here? Read [TEAM_HANDOFF.md](TEAM_HANDOFF.md) first** (who owns what, getting data onto TC2, how to plug in your own angle).

AI6103 group project, Route 2 (improve a 2019+ paper). Paper: HiFT, Hierarchical Feature
Transformer for Aerial Tracking (Cao et al., ICCV 2021), official code pinned at commit `992aa7c`.

## In plain words

HiFT follows one object through drone video, fast enough to run on the drone. We ask two questions:

1. **Adversarial track.** Can tiny, invisible pixel changes make HiFT lose its target, and can a
   cheap image clean-up step (blur, JPEG, fewer colour levels, median filter) stop that without
   retraining?
2. **Edge-cost track.** How small and fast can HiFT get with no retraining (half precision, int8,
   pruning), how much accuracy does that cost, and does the compressed model stay just as easy
   to fool?

The second half of question 2 links the two tracks: the attack is computed on the normal model
and replayed against the compressed ones.

## What is in here

| Path | What it does |
|---|---|
| `tools/patch_hift.py` | Makes the official code run on CPU / Apple MPS / CUDA (removes 4 hard-coded `.cuda()` calls; identical behaviour on CUDA) |
| `hiftstress/tracker.py` | The official HiFT tracker, with a hook on every 287x287 search crop: attack, then defence, then any model variant |
| `hiftstress/attacks.py` | Random-sign noise (control), FGSM, PGD; objective = lower HiFT's target score near the crop centre (the idea behind the Cooling-Shrinking Attack, CVPR 2020) |
| `hiftstress/defenses.py` | Gaussian blur, median filter, bit-depth reduction, JPEG; each usable in an adaptive attack (exact gradient or BPDA) |
| `hiftstress/edge.py` | Variants fp32, fp16, int8dyn, int8bb (int8 backbone, calibrated), prune30/50/70, compile; latency / FLOPs / size benchmark |
| `hiftstress/metrics.py` | OPE success AUC and precision@20 px, same conventions as the pysot toolkit HiFT reports with |
| `scripts/run_track.py` | One condition (variant x attack x defence) over a benchmark -> `results/<run>/summary.json` + per-sequence results + pysot-format boxes |
| `scripts/bench_edge.py` | Size, GFLOPs, latency (mean, p90, backbone vs head), GPU memory, and score-map fidelity vs fp32 |
| `scripts/sweep_adv.sh`, `scripts/sweep_edge.sh` | The full grids; resumable (finished runs are skipped) |
| `scripts/summarize.py` | All runs -> `results/tables/*.csv`, `summary.md`, and figures |
| `scripts/smoke_test.py` | 13 pipeline checks, no data or weights needed |
| `slurm/` | TC2 jobs (same partition/QoS/env pattern as the dog-breed study) |

## Benchmark: UAV123 (data lives on the cluster)

Main benchmark: **UAV123** (123 drone videos, about 113k frames, 30 fps). HiFT reports results on
it. The official KAUST page no longer resolves and DTB70 is only on Baidu, so UAV123 comes from a
Hugging Face mirror of the KAUST release (`xche32/UAV123`, 14 GB). The cluster downloads it
directly, so it never touches your laptop. Frame ranges for the split sequences (bird1_1, bird1_2,
...) are bundled from the GOT-10k toolkit (`tools/ranges/`, MIT licence). UAV20L, the long-term
subset, comes in the same archive.

If you do get DTB70 later, put it at `data/DTB70/<seq>/img/*.jpg` + `groundtruth_rect.txt` and set
`DATASET=dtb70 ROOT=data/DTB70 TAG=dtb70`.

## Step 1: laptop setup (about 10 minutes, no dataset needed)

```bash
cd hift-stress
bash setup_local.sh        # venv, HiFT code, patch, 13 checks with random weights
cp ~/Downloads/first.pth weights/general_model.pth     # the official HiFT weights
source .venv/bin/activate
python scripts/smoke_test.py --snapshot weights/general_model.pth   # expect 13/13
```

## Step 2: TC2 cluster

Connect GlobalProtect VPN, then from the laptop:

```bash
rsync -av --exclude .venv --exclude results --exclude third_party ./ <you>@10.96.189.12:~/hift-stress/
```

(this sends `weights/` too). On the login node, one job at a time, checking each log before the next:

```bash
cd ~/hift-stress && mkdir -p runs
sbatch slurm/job_setup_tc2.sbatch          # once: env + torch cu128 + HiFT patch (about 15 min)
sbatch slurm/job_data_uav123_tc2.sbatch    # once: download 14 GB, unpack, convert (no GPU)
sbatch slurm/job_smoke_tc2.sbatch          # checks on GPU + minutes per full-UAV123 run
```

What to check:
- `runs/data_*.out` ends with `data/uav123_flat: 123 sequences, ~112578 frames`. If fewer, the
  skipped list above it says why.
- `runs/smoke_*.out`: 13/13 checks pass, clean AUC on the 5-sequence subset is sensible (compare
  with HiFT's UAV123 row in the paper's results table), PGD is clearly lower, and the minutes-per-run estimate.

If one run would not fit the 6 h wall (PGD-10 on all 113k frames is the slowest), lower `STEPS` or
use a fixed subset for the attack grid, for example `MAXSEQ=40` in the `--export` list, and say so
in the report. Then submit, each independent, from the login node, never from inside a job:

```bash
sbatch --export=ALL,PHASE=A slurm/job_adv_tc2.sbatch
sbatch --export=ALL,PHASE=B slurm/job_adv_tc2.sbatch
sbatch --export=ALL,PHASE=C slurm/job_adv_tc2.sbatch
sbatch --export=ALL,PART=gpu slurm/job_edge_tc2.sbatch
sbatch --export=ALL,PART=cpu slurm/job_edge_tc2.sbatch
```

A job that hits the wall time can be resubmitted unchanged: finished runs are skipped. Bring
everything back with `rsync -av <you>@10.96.189.12:~/hift-stress/{results,runs} ./`.

## Optional: quick runs on the laptop

The data job also writes `data/uav123_mini.tar` (5 real sequences x 60 frames, a few MB). Copy it
back and unpack it to try things locally. Laptop numbers are **not** results.

```bash
scp <you>@10.96.189.12:~/hift-stress/data/uav123_mini.tar data/ && tar -xf data/uav123_mini.tar -C data/
source .venv/bin/activate
export DATASET=flat ROOT=data/uav123_mini TAG=mini SNAP=weights/general_model.pth MAXSEQ= MAXFRAMES= STEPS=5 PY=python
PHASE=A bash scripts/sweep_adv.sh
PART=cpu ITERS=50 bash scripts/sweep_edge.sh
python scripts/summarize.py --results results && open results/tables/summary.md
```

If anything fails on Apple MPS, add `DEVICE=cpu`. Delete these results (`rm -rf results`) before
copying cluster results back, so they never mix.

## The experiment grid

**Adversarial track** (fp32 model, full UAV123, eps in pixels out of 255)

| Phase | Conditions | Question it answers |
|---|---|---|
| A | clean; noise 2/4/8; FGSM 1/2/4/8; PGD-10 1/2/4/8 | How fragile is HiFT, and is it the attack or just noise? |
| B | blur1, median3, bits5, jpeg75 x {clean, PGD-10 eps 4, adaptive PGD-10 eps 4} | Which defence buys robustness at what clean cost, and does it survive an attacker who knows about it? |
| C | PGD eps 4 (from fp32) vs fp16, prune30, prune50, int8bb | Does compression change robustness? |

**Edge track**

| Part | Variants | Measured |
|---|---|---|
| gpu | fp32, fp16, prune30/50/70, compile | latency mean/p90, backbone vs head, peak memory, size, GFLOPs, fidelity, clean AUC |
| cpu | fp32, int8dyn, int8bb, prune50 at 1/2/4 threads | the same on a CPU (a stand-in for an onboard computer; report the CPU model from the log) |

## Rules that keep the results honest

- Report only runs with real weights; `summarize.py` warns about random-weight runs.
- Every attack row needs the noise row with the same eps next to it.
- A defence only counts if it holds under the **adaptive** attack (Athalye et al., ICML 2018).
- State the threat model: white-box, digital, per-frame perturbation of the search crop.
- Pruned models are not faster on dense hardware; say so and report `nonzero_params_m` instead of claiming speed.
- Name the hardware (GPU model, CPU model, threads) for every latency number; TC2 is not a drone.
- Time the attack separately (`ms_per_frame.attack`): an attack slower than the camera frame rate is an offline upper bound, not a real-time threat.
- One seed by default. If two conditions differ by less than about 1 AUC point, rerun with `SEED=1` and `SEED=2` before claiming a difference.

## Known limits

- No retraining. Adversarial training and quantisation-aware training need the 4 large training sets (VID, COCO, GOT-10k, YouTube-BB); left as future work.
- int8 variants run on CPU only (PyTorch eager quantisation); fp16 needs a GPU or Apple MPS.
- The attack perturbs the crop the network sees, not the physical scene.
