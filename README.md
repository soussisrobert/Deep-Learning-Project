# Stress-Testing HiFT for Aerial Tracking: Low Light, CPU Inference and Domain Shift

AI6103 Deep Learning & Applications (NTU), group project, Route 2: we improve the official
[HiFT](https://github.com/vision4robotics/HiFT) tracker (Cao et al., ICCV 2021) along three separately
evaluated tracks, each against its own matched baseline:

| Track | Change | Main result (success AUC) | Folder |
|---|---|---|---|
| Night | gamma 0.6 on template/search crops (+ anchored template) | UAVDark135 pooled-39: 0.380 → 0.451, +0.071 [+0.028, +0.116] | [`tracks/night`](tracks/night) |
| CPU efficiency | static int8 backbone, `layer1` and head kept fp32 (int8bb-s1) | 1.75× network, 1.72× tracker step on Xeon Gold 6230R; UAV123 TEST-40 −0.0074 [−0.019, −0.001] | [`tracks/cpu_int8`](tracks/cpu_int8) |
| Domain fine-tuning | REF recipe on VisDrone + UAVDT (corrected layout/losses, B2-L1 head) | VisDrone-34: +0.0749 [+0.024, +0.124]; UAV123@10fps-76: −0.0050 [−0.034, +0.027] | [`tracks/domain_ft`](tracks/domain_ft) |

All tracks start from the released HiFT checkpoint (SHA-256 `8c697d2b…ba3101`) and HiFT commit
`992aa7c1605360d607af84cb804b93cd8ee81caf`. The study concerns recorded aerial video, not drone control.

## Quick start: reproduce every number in the paper (2 minutes, no GPU, no datasets)

```bash
git clone https://github.com/soussisrobert/Deep-Learning-Project.git
cd Deep-Learning-Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r environment/requirements-verify.txt
python scripts/verify_paper_numbers.py
```

Expected last line: `207/207 paper numbers reproduced`. The script recomputes every mean and every
bootstrap interval from the released per-video files in [`results/`](results) using one shared
statistics module ([`shared/stats.py`](shared/stats.py)), and checks each against
[`paper/expected_numbers.json`](paper/expected_numbers.json) (the value as printed, plus where it appears).
The same check runs on every push ([`.github/workflows/verify.yml`](.github/workflows/verify.yml)).

Deeper levels (re-scoring saved boxes, re-running tracking, retraining) are in [REPRODUCE.md](REPRODUCE.md).

## Repository layout

```
README.md                 this page
REPRODUCE.md              level 1 / 2 / 3 reproduction, exact commands per track
environment/              pinned requirements (verification, and per-track runtime)
shared/stats.py           the one bootstrap implementation used for every interval
scripts/
  verify_paper_numbers.py level-1 check of all paper numbers
  check_evidence_checksums.py  integrity check of the domain evidence bundle
  convert/                how results/ was produced from each track's raw outputs
results/                  per-video scores of every model on every benchmark, common format
  night/ cpu/ domain/ box_refinement/ crop_zoom/   common format: benchmark,video,frames,model,seed,success_auc,precision20,toolkit
  numbers.json            every computed number (written by verify --write)
paper/expected_numbers.json   the numbers as printed in the paper
data/                     how to obtain each dataset; data/splits/ = every sequence list used
checkpoints/              where to get each checkpoint, with SHA-256 (weights are not in git)
tracks/
  night/                  HiFT fork with gamma + anchored template
  cpu_int8/               hift-stress: int8 CPU inference, timing, robustness
  domain_ft/              DroneUse: domain fine-tuning REF and its sealed evaluation
```

Each track folder keeps its own history, logs and reports unchanged; `TRACK.md` in each one is the
short guide (what it changes, which paper section it supports, how to run it).

## Claims table

| Paper claim | Value | Source file | Reproduce |
|---|---|---|---|
| Gamma on UAVDark135 pooled-39 (external) | +0.071 [+0.028, +0.116] | `results/night/per_video_UAVDark135_{test,val}.csv` | level 1 |
| NAT2021 original → gamma → gamma+anchor (selected here) | 0.372 → 0.417 → 0.453 | `results/night/per_video_NAT2021.csv` | level 1 |
| Anchor increment on UAVDark135 pooled-39 | +0.013 [−0.031, +0.058] | same as above | level 1 |
| Original HiFT vs published NAT2021 | 0.372 / 0.553 vs 0.370 / 0.545 | `results/night/per_video_NAT2021.csv` | level 1 |
| No NAT2021 frame in UAVDark135 (sampled screen) | 0 / 2,578 | `tracks/night/evidence/sources/output_HiFT_uavdark_leak_49103.out` | recorded |
| Losses persist: lost frames in ≥101-frame stretches | 86.3% (43,793 / 50,749) | `tracks/night/evidence/sources/E15_*.csv` | level 1 |
| int8bb-s1 network / tracker-step speed-up | 1.75× / 1.72× | `tracks/cpu_int8/analysis/2026-11-02_retime-matched/retime_matched_t4/summary.json` | recorded |
| TEST-40 quality change (two-level interval) | −0.0074 [−0.019, −0.001] | `results/cpu/per_video_UAV123.csv` | level 1 |
| All-123 quality change | +0.0030 [−0.0054, +0.0140] | `results/cpu/per_video_UAV123.csv` | level 1 |
| PGD-4 attack on GPU fp32 | 0.5825 → 0.0955 | `results/cpu/per_video_UAV123.csv` | level 1 |
| Confidence detects failure / warns early | AUROC 0.968; 10 / 32 onsets | `tracks/cpu_int8/analysis/2026-10-22_round5-final/numbers5.json` | recorded |
| REF vs B0, VisDrone-34 (22 source clusters) | +0.0749 [+0.024, +0.124] | `results/domain/per_video_VisDrone34.csv` | level 1 |
| REF vs B0, UAV123@10fps-76 (48 clusters) | −0.0050 [−0.034, +0.027] | `results/domain/per_video_UAV123_10fps_76.csv` | level 1 |
| Box refinement vs stock HiFT, VisDrone-34 (exploratory) | +0.0156 [+0.0053, +0.0280] | `results/box_refinement/per_video_VisDrone34.csv` | level 1 |
| Crop zoom k = 0.9 vs native, UAV123 10 fps subsample | −0.248 [−0.288, −0.207] | `results/crop_zoom/per_video_crop_zoom.csv` | level 1 |

*Level 1* = recomputed from released files by `verify_paper_numbers.py`. *Recorded* = read from a
released measurement file (timing logs, frame-level analysis output); recomputing it needs raw tracker
outputs or hardware (level 2/3). The full list of 207 checked numbers is `paper/expected_numbers.json`.

## What is not in git

Model weights and datasets. See [`checkpoints/README.md`](checkpoints/README.md) (download links,
SHA-256 values, release assets) and [`data/README.md`](data/README.md) (official dataset sources and
the exact sequence lists).

## Authors and contributions

| Author | Part of the paper | Folder |
|---|---|---|
| Ayush Bhatt | Night tracking: gamma 0.6, anchored template, NAT2021/UAVDark135 evaluation, loss-stretch and overlap analyses | `tracks/night` |
| Manzi Roger Emile | CPU efficiency: int8 recipe, matched timing, quality analysis, confidence and attack diagnostics; repository and shared statistics | `tracks/cpu_int8`, `shared/`, `scripts/` |
| Bao Ning | Domain fine-tuning: REF recipe, code corrections, sealed evaluation | `tracks/domain_ft` |
| Soussis Robert | Output-only corner-head box refinement | `tracks/box_refinement` |
| Aayushi Priyam | Crop-scale and interpolation check | `tracks/crop_zoom` |

## Licence

HiFT and pysot code inside `tracks/night` (and fetched by `tracks/cpu_int8`) is under its original
Apache-2.0 licence ([tracks/night/LICENSE](tracks/night/LICENSE)). Datasets are not redistributed and
keep their own licences.
