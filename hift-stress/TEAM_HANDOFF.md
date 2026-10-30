# HiFT on TC2: shared setup + team handoff

AI6103 group project, Route 2: improve HiFT (Cao et al., ICCV 2021, real-time drone tracker).

**All results on one page: [RESULTS.md](RESULTS.md).**

This repo is the **shared base** everyone can build on. It gives you working HiFT code, weights, the UAV123 benchmark on TC2, and an evaluation harness.
Each member picks their own improvement angle on top of it.

## Round 5 (Manzi's tracks, complete)
- **Results:** `analysis/2026-10-22_round5-final/REPORT.md`.
  - int8bb-s1 holds accuracy on all 123 videos: about −0.004 once two fragile videos are set aside.
  - HiFT's confidence detects natural failures (AUROC 0.97) but not attacks.
  - int8 is as robust as fp32 under 7 corruptions.

- **What it is:** closes round 4 (int8 recipe on the 83 non-TEST videos, plus official attribute labels). It also tests whether HiFT's own confidence warns of failures and attacks, including an attacker who tries to stay confident, and natural corruptions including synthetic night.
- **Plan:** [ROUND5_PLAN.md](ROUND5_PLAN.md).
- **Agent instructions:** [AGENT_RUNBOOK_ROUND5.md](AGENT_RUNBOOK_ROUND5.md).
- **Round-4 results:** `analysis/2026-10-21_round4-final/REPORT.md`.

## Round 4 (Manzi's tracks, complete except J1c, which is deliberately not run)
- **What it is:** gated follow-up to round 3. It tests JPEG-85 on TEST and three new int8 calibration recipes (DEV pilot first). A combined branch runs only if both pass.
- **How it runs:** `scripts/r4_gate.py` decides what runs next and cuts branches that fail.
- **Plan:** [ROUND4_PLAN.md](ROUND4_PLAN.md).
- **Agent instructions:** [AGENT_RUNBOOK_ROUND4.md](AGENT_RUNBOOK_ROUND4.md).
- **Round-3 results:** `analysis/2026-10-21_round3-final/REPORT.md`.

## Round 3 (Manzi's tracks, complete)
- **What it is:** stress-tests the two round-2 near-misses (the JPEG filter and the int8 backbone) against stronger attackers. The JPEG quality is chosen on 20 held-out videos. It also tests GPU-side JPEG, int8 + JPEG together on CPU, int8 calibration spread and natural corruptions.
- **Plan and pass rules:** [ROUND3_PLAN.md](ROUND3_PLAN.md).
- **Agent instructions:** [AGENT_RUNBOOK_ROUND3.md](AGENT_RUNBOOK_ROUND3.md).
- **Queues:** [plan/round3/SCHEDULE.md](plan/round3/SCHEDULE.md).
- **Round-2 results:** `analysis/2026-10-19_round2-final/REPORT.md`.

## Round 2 (Manzi's tracks, complete)
**Running it on TC2 (agent instructions): [AGENT_RUNBOOK_ROUND2.md](AGENT_RUNBOOK_ROUND2.md).** See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) (hypotheses, pass rules, literature) and [plan/SCHEDULE.md](plan/SCHEDULE.md) (TC2 job queues). Round-1 analysis: `analysis/2026-10-17_first-results/REPORT.md`.

## Who owns what
| Part | Owner | Status |
|---|---|---|
| Shared base: HiFT runs anywhere, weights, UAV123 on TC2, metrics, job scripts | Manzi (built, use freely) | Verified on CPU; not yet run on TC2 GPU |
| Adversarial robustness (attacks + input defences) | Manzi | Code done; full runs pending |
| Edge cost (fp16 / int8 / pruning: speed vs accuracy) | Manzi | Code done; full runs pending |
| Other angles (see "Ideas" below) | Open: pick one | Not started |

Please don't change `hiftstress/attacks.py`, `defenses.py` or `edge.py` without telling Manzi; those are her results.
Add your own module or script instead.

## What is in the repo
- `tools/patch_hift.py`: clones official HiFT at commit `992aa7c` and makes it run on CPU, Mac or GPU (removes 4 hard-coded `.cuda()` calls; no other change).
- `hiftstress/`: tracker wrapper, data loaders, OPE metrics (success AUC and precision at 20 px, pysot conventions).
- `scripts/run_track.py`: runs one condition over a benchmark and writes `results/<run>/summary.json` + per-sequence results + boxes.
- `scripts/smoke_test.py`: pipeline checks (17 on CPU, 18 on a GPU, which adds fp16).
- `scripts/summarize.py`: turns results into tables and figures.
- `slurm/`: TC2 jobs (partition MGPU-TC2, 1 GPU, 6 h, private conda env in `~/hift-stress/env`).

**Not in the repo (too big or not ours):**
- Weights: get `general_model.pth` from the Google Drive link in [HiFT's README](https://github.com/vision4robotics/HiFT) and save it as `weights/general_model.pth`. It's 86 MB, epoch 67, and all keys load.
- Data: downloaded on TC2 (next section).

## Getting the data onto TC2 (no laptop download)
- The KAUST UAV123 page is dead, and DTB70 is only on Baidu.
- We use the public Hugging Face mirror `xche32/UAV123`: one 14 GB `UAV123.tar.gz` in the KAUST folder layout. No login is needed.
- `slurm/job_data_uav123_tc2.sbatch` downloads it with `huggingface_hub` (resumable), unpacks it, and converts it with `scripts/convert_uav123.py`.
- The split sequences (e.g. bird1_1, bird1_2) use frame ranges bundled from the GOT-10k toolkit (MIT licence).
- Expect `data/uav123_flat: 123 sequences, 112578 frames`. You need about 30 GB free while it unpacks.
- Don't use the `load_dataset(...)` snippet shown on the Hugging Face page. It only reads text files, not the video frames.

## Replicate
```bash
# laptop
git clone <this repo> hift-stress && cd hift-stress
bash setup_local.sh                                   # venv + HiFT + patch + 17 checks (random weights)
# put general_model.pth in weights/, then:
.venv/bin/python scripts/smoke_test.py --snapshot weights/general_model.pth    # expect 17/17
rsync -av --exclude .venv --exclude third_party ./ <you>@10.96.189.12:~/hift-stress/   # VPN on

# TC2 login node, one job at a time; read each log before the next
mkdir -p runs logs
sbatch slurm/job_setup_tc2.sbatch          # env + torch cu128 + HiFT (about 15 min)
sbatch slurm/job_data_uav123_tc2.sbatch    # UAV123 download + convert
sbatch slurm/job_smoke_tc2.sbatch          # GPU checks + clean vs PGD on 5 seqs + minutes per full run
```
TC2 rules:
- Run `sbatch` from the login node only. Don't chain `sbatch` inside a job; TC2 refuses it.
- Don't run compute on the login node.
- Limits: 1 GPU, 6 h per job, 2 jobs at a time.

**Clean baseline for everyone:**
```bash
python scripts/run_track.py --dataset flat --root data/uav123_flat --snapshot weights/general_model.pth --device cuda
```
Compare its AUC with HiFT's UAV123 row in the paper before trusting anything else.

## Status on TC2 (round 1 done, 17 Oct)

**The setup works on TC2.**

| Item | Value |
|---|---|
| GPU | NVIDIA A40 |
| Python | 3.11.17 |
| PyTorch | 2.8.0+cu128 |
| HiFT | pinned at `992aa7c`, with the inference device patches |
| Smoke test | 14/14 checks passed |
| Data | UAV123 unpacked to 123 sequences, 112,578 frames |

- Exact package versions: `environment-tested.txt`.
- Checkpoint checksum: `checkpoint.sha256`.
- The fixed 40-video attack list: `attack-subset-40.json`.

**Round-1 results.** Full tables are in `verified-results/` (copied from TC2). The analysis is in `analysis/2026-10-17_first-results/REPORT.md`.

On the 40 videos, ε = 4/255:

| Condition | Success AUC | Precision@20 (our strict rule) |
|---|---|---|
| Clean | 0.582 | 0.754 |
| Random noise (same size) | 0.563 | 0.729 |
| FGSM (1 step) | 0.418 | 0.672 |
| PGD (10 steps) | **0.095** | 0.323 |

- **Targeted attacks hurt far more than random noise:** PGD removes about 84% of the success score.
- **Clean results:** all 123 videos on GPU give AUC 0.584; on CPU with 4 threads, 0.586. HiFT's paper reports 0.589.
- **Edge cost:** the fp32 network takes 89 ms on 4 Xeon threads. With int8 in the backbone (`int8bb`) it takes 43 ms (2.1× faster) and the model shrinks from 44 to 33 MB. `prune50` gives no speed-up.

**Precision note:** our 0.754 is below the paper's 0.787 because our rule counts frames where the target is absent as misses. HiFT's toolkit counts them as hits. Round 2 reports both.

**Practical lessons from TC2**
- **Job limits:** at most 2 submitted jobs at once, and 6 h per job.
- **Resubmitting:** completed conditions are skipped. An interrupted condition restarts from the beginning.
- **CPU-side attacks are too slow for one job:** CPU PGD on 40 videos takes about 22 h. Round 2 avoids this by computing the attack on the GPU (queue 09).

**Still unverified:**
- A fresh install from the zip on a new machine.
- Mac / Apple MPS.

**What round 1 left open is now queued in round 2** (`AGENT_RUNBOOK_ROUND2.md`):

| Open from round 1 | Round-2 queues |
|---|---|
| Defences, with attacks that know the defence | P1 |
| Full accuracy of the compressed models | P2, P5 |
| Other PGD strengths, plus seeds | P3 |
| Compressed models under attack, with GPU-side attack splitting | P4 |

## Ideas for your own angle (each grounded in a known HiFT limitation)
- **Night / low light:** HiFT was trained on daytime data. Test on NAT2021 (UDAT, CVPR 2022, public) and try low-light augmentation or fine-tuning.
- **Occlusion / long-term memory:** HiFT matches each frame against the first-frame template only. Try a learned template update (UpdateNet, ICCV 2019) and evaluate on UAV20L, which `convert_uav123.py` already builds as `data/uav20l_flat`.
- **Knowing when it fails:** turn the score map into a confidence score (MC dropout, ensembles, temperature scaling) and measure how well it predicts tracking failure.
- **Backbone swap:** HiFT's AlexNet lacks global context. Try a small ResNet, MobileNet or ViT and plot accuracy against speed.

## How to plug in your own method
- **Change what the network sees each frame:** write a callable `x -> x` on the 1×3×287×287 crop (0–255 BGR) and pass it as `defense=` to `TrackingModel` (see `hiftstress/tracker.py`).
- **Change the model:** build it the way `make_variant` in `hiftstress/edge.py` does, then pass it as `net`.
- **New dataset:** add a loader in `hiftstress/data.py`. It must return `Sequence(name, frames, gt)`.
- **Reporting:** every result is a folder with `summary.json`. Put your runs under your own `--out` folder (e.g. `results_<name>/`) so we don't overwrite each other.
