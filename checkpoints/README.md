# Checkpoints

Weights are not stored in git. Download them, put them where each track expects them, and check the
SHA-256 with `sha256sum -c checkpoints/SHA256SUMS` (paths in that file are relative to `checkpoints/`).

| File | What | Where to get it | Used by |
|---|---|---|---|
| `general_model.pth` | official HiFT checkpoint (also distributed as `first.pth`) | links in [the HiFT README](https://github.com/vision4robotics/HiFT#2-test) | all tracks (baseline and starting point) |
| `REF_seed20261023.pt` | domain fine-tuned REF, seed 20261023 | from the authors on request (SHA-256 in `SHA256SUMS`) | domain track |
| `REF_seed20261102.pt` | domain fine-tuned REF, seed 20261102 | from the authors on request (SHA-256 in `SHA256SUMS`) | domain track |
| `corner-head.pt`, `coarse-refiner.pt`, `warmup-corner-head.pt` | box refinement heads | GitHub Release `v1.0-paper` assets; hashes in `tracks/box_refinement/handoff/SHA256SUMS` | box refinement |
| `alexnet-bn.pth` | ImageNet AlexNet backbone for HiFT training | upstream HiFT repository, `pretrained_models/alexnet-bn.pth`; goes in `tracks/night/pretrained_models/` | night fine-tuning only |

The int8bb-s1 model is not a separate download: it is rebuilt deterministically from
`general_model.pth` and the calibration list recorded in
`tracks/cpu_int8/analysis/2026-11-02_retime-matched/retime_matched_t4/summary.json`.

Where each track looks for the official checkpoint:
- night: `tracks/night/tools/snapshot/first.pth`
- CPU: `tracks/cpu_int8/weights/general_model.pth`
- domain: `tracks/domain_ft/evidence/stage7/artifacts/checkpoints/general_model.pth`
  (REF: `.../artifacts/tc2-runs/TC2-P1-REF-001/final-model.pt` and `.../TC2-P1-REF-S2-001/final-model.pt`)

Raw per-frame CPU-track outputs for level-2 re-scoring are available from the authors on request.
Every number in the paper is reproducible from the released per-video files without them (level 1).
