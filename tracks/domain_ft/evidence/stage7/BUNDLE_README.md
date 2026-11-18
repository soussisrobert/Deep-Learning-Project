# HiFT Stage 7 evidence bundle

This archive preserves the workspace-relative paths and byte contents of the original records. No training, inference, or new benchmark evaluation was run to make it. No source frame images or training datasets are included.

## Start here

- `reports/STAGE7_CLOSEOUT.md` and `artifacts/stage7/closeout.json`: frozen REF decision, stored G1–G4 gate, and cluster bootstrap results.
- `artifacts/b2-eval/STAGE2-SEALED-*`: six evaluation JSONs, six inference manifests, and 333 prediction CSVs (three runs per dataset; 35 VisDrone or 76 UAV123 sequences per run).
- The two sealed dataset manifests contain ordered frame identities and ground truth. `source_cluster_mapping.csv` is a new 111-row export of the original Stage 7 grouping rule, not a historical artifact. VisDrone has 23 clusters, or 22 after excluding `uav0000079_02568_s` from the disjoint-34 set; UAV123 has 48 name-derived clusters.
- `artifacts/tc2-runs/TC2-P1-REF-001/` and `TC2-P1-REF-S2-001/` (under the same parent): both REF checkpoints and their training manifests. The official B0 checkpoint is included to make the Stage 7 checkpoint hash gate independently checkable. The evaluator hashes the checkpoint files without loading them.
- `reports/STAGE[2-6]_PREREGISTRATION.md`, `artifacts/tc2-runs/`, `artifacts/tc2-validation/`, and the stage decision JSONs: per-arm validation scores and gate outcomes, including `configs/stage3_track_grid.json` and all 111 model-setting validation records.
- `artifacts/tc2-runs/TC2-S4-REFC-*/validation/`: per-frame boxes and scores for both REF-C seeds, 131 episodes per seed. The validation annotation snapshots used in stages 3–5 are included under `artifacts/stage3-diag/x/data/` without images.

## Gaps and protocol notes

- The exact Stage 7 REF-C loss/recovery rule, event list, wrong-object matches, and original review scripts were not located in either checkout. The report says the scratch scripts stayed in the Windows temp folder. The earlier Stage 3 abrupt-loss definition is not a substitute for the exact Stage 7 protocol.
- Both sealed evaluations use stride 1 over their supplied frame lists. UAV123 uses the publisher-supplied 10 fps variant of the 30 fps original (nominal 3:1 downsampling); the evaluation does not apply an extra stride of 3. The bundled original README and `configSeqs.m` document the variant and ranges. VisDrone capture FPS is not verified in either checkout.
- Local source checkout `c8c69b857d19a0a039ffdf8b7f9bc2ae3b164764`; secondary checkout `/mnt/Study/Code/HiFT` at `8687145794b0692d9dae6ec3a433a1b4b0c99fe0`. The latter does not have `artifacts/stage7/closeout.json`. The two dataset manifests and all stage 3–6 decision JSONs match byte for byte. Core sealed evidence and checkpoint comparison: 350/350 checked files match byte for byte.

## Verification

From the extracted bundle root, run `sha256sum -c SHA256SUMS` to verify every listed file. The two REF digests are listed in full. To reproduce the CPU-only closeout in a disposable copy, preserve or remove the existing `artifacts/stage7/closeout.json` from that copy first (the script refuses to overwrite), then run `python scripts/stage7_closeout.py` with NumPy installed. The stage 3–6 scripts may overwrite their decision JSONs, so run them only in a disposable copy as well.
