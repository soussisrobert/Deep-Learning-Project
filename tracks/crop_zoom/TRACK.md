# Crop-zoom track: test-time crop scale and interpolation (negative result)

**Paper section:** Night tracking, paragraph "Crop scale and interpolation".
**Base:** upstream HiFT `main` (vision4robotics/HiFT), official checkpoint `first.pth`; Kaggle, Tesla T4.

## What changes (test-time only, no retraining)
Patches applied inside the notebooks (`notebooks/nb2.ipynb`, cell 7):
- `TRACK.ZOOM = k`: the template and search crop side lengths are divided by k (`hift_tracker.py`, in
  `init` and `track`). Because `scale_z`, the scale penalty and the box mapping all use the zoomed crop,
  boxes are returned in image pixels consistently; k = 1.0 is the original tracker.
- `TRACK.INTERP`: the crop resize uses linear (default), cubic, Lanczos, area, or `auto`
  (cubic when enlarging, area when shrinking) (`base_tracker.py`).

Settings run: UAV123@10fps k ∈ {0.80, 0.90, 1.00, 1.15, 1.30, 1.50} and `auto` at k = 1.0;
UAV20L k ∈ {0.90, 1.00, 1.30, 1.50} and `auto` at k = 1.0. All 123 / 20 videos.

## Data
- UAV123@10fps here is built from the 30 fps UAV123 release by keeping every third frame and every
  third annotation (`notebooks/nb1.ipynb`, cell 8); the official 10 fps download failed. It is not the
  domain track's official UAV123@10fps release.
- UAV20L: the 20 long UAV123 sequences, every frame (30 fps).

## Scoring
`export_per_video.py` re-scores every run with the paper's rule (missing/invalid ground truth counts as a
failure) and writes `../../results/crop_zoom/per_video_crop_zoom.csv`. Intervals: paired video bootstrap,
10,000 draws, seed 20261121 (`scripts/verify_paper_numbers.py`).
