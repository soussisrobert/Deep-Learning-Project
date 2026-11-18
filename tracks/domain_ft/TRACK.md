# Domain fine-tuning track: REF (VisDrone + UAVDT)

**Paper section:** Domain fine-tuning; Table 3.
**Base:** HiFT commit `992aa7c`, official checkpoint (SHA-256 `8c697d2b…ba3101`).
`README.md` in this folder is the original project README, kept unchanged.

## What changes
REF fine-tunes backbone layers 3–5 and the complete head on VisDrone MOT-train + SOT train_part1 and
UAVDT (minus a 10% video holdout), with a corrected encoder tensor layout, masked target/loss
computation and an added unit-weight L1 regression loss (B2-L1). Training: 10 epochs × 1,000 updates,
batch 32, head learning rate 1e-3 → 1e-4 (log-linear), seeds 20261023 and 20261102
(`evidence/stage7/artifacts/tc2-runs/TC2-P1-REF-*/manifest.json`). B0 = official weights with the same
sigmoid scores and NaN-safe selection.

## Selection vs report
Eight training arms and a 37-setting post-processing grid were compared on validation
(`evidence/stage7/artifacts/tc2-validation/`, `evidence/stage7/configs/stage3_track_grid.json`, stage decisions in
`evidence/stage7/artifacts/stage3-diag … stage6/`); none was adopted over REF. The sealed test sets
(VisDrone test-dev disjoint-34, UAV123@10fps-76) were scored once (stage 2) and closed out in stage 7.

## Evidence
- `evidence/stage7/`: frozen bundle as delivered, with its own `BUNDLE_README.md` and `SHA256SUMS`
  (`python scripts/check_evidence_checksums.py` from the repository root).
- `reports/STAGE7_CLOSEOUT.md`: the close-out report.
- Per-video results in the common format: `../../results/domain/`.

## Not included
The REF-C loss/recovery review (event list and scripts were not retained); weights (see
`../../checkpoints/README.md`).
