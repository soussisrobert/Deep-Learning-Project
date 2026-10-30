# HiFT on TC2: shared setup + team handoff

AI6103 group project, Route 2: improve HiFT (Cao et al., ICCV 2021, real-time drone tracker).
This repo is the **shared base** everyone can build on. It gives you working HiFT code, weights, the UAV123 benchmark on TC2, and an evaluation harness.
Each member picks their own improvement angle on top of it.

## Who owns what
| Part | Owner | Status |
|---|---|---|
| Shared base: HiFT runs anywhere, weights, UAV123 on TC2, metrics, job scripts | Sandhy (built, use freely) | Verified on CPU; not yet run on TC2 GPU |
| Adversarial robustness (attacks + input defences) | Sandhy | Code done; full runs pending |
| Edge cost (fp16 / int8 / pruning: speed vs accuracy) | Sandhy | Code done; full runs pending |
| Other angles (see "Ideas" below) | Open: pick one | Not started |

Please don't change `hiftstress/attacks.py`, `defenses.py` or `edge.py` without telling Sandhy; those are her results.
Add your own module or script instead.

## What is in the repo
- `tools/patch_hift.py`: clones official HiFT at commit `992aa7c` and makes it run on CPU, Mac or GPU (removes 4 hard-coded `.cuda()` calls; no other change).
- `hiftstress/`: tracker wrapper, data loaders, OPE metrics (success AUC and precision at 20 px, pysot conventions).
- `scripts/run_track.py`: runs one condition over a benchmark and writes `results/<run>/summary.json` + per-sequence results + boxes.
- `scripts/smoke_test.py`: 13 pipeline checks.
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
bash setup_local.sh                                   # venv + HiFT + patch + 13 checks (random weights)
# put general_model.pth in weights/, then:
.venv/bin/python scripts/smoke_test.py --snapshot weights/general_model.pth    # expect 13/13
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

## Verified so far
Verified in a cloud sandbox, on CPU only:
- 13/13 checks pass with the real weights, from a fresh install.
- A synthetic-video preview shows the pipeline behaves as expected. These are not results.

**Not done yet:**
- Any TC2 GPU run.
- Confirming the mirror unpacks to 123 sequences.
- Real timings, including whether full-UAV123 runs fit 6 h. If not, use `--max-seqs N` and say so in the report.
- Apple MPS.

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
