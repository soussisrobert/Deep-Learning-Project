# Box refinement track: corner-head refinement of HiFT's output box (exploratory)

**Paper section:** Domain fine-tuning, paragraph on box refinement; Table 3.
**Owner's package:** `handoff/` (as delivered; `README.txt`, `Robert reply.txt`). Weights are release assets.

Stock HiFT (original weights, raw scores) → coarse refiner proposal → corner head on a 2.5x crop; if
visibility >= 0.65 the box moves 35% toward the corner estimate. Output-only: HiFT's own state drives the next
search. Trained on 86 VisDrone-SOT train clips (66 source IDs), seed 0; no source shared with val (11) or
test (35) clips (`handoff/dataset_inventory.csv`).

Result on VisDrone test-dev disjoint-34 (22 source clusters), domain evaluator: 0.5300 → 0.5456,
+0.0156 [+0.0053, +0.0280] (`../../results/box_refinement/per_video_VisDrone34.csv`, checked by
`scripts/verify_paper_numbers.py`).

Caveats: one seed; an earlier refiner version was scored on the same test clips; on validation the full
pipeline (0.468) scored below the coarse refiner alone (0.473); code snapshot made after the experiments.
