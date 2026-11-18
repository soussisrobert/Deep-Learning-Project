# Experimental identities and evidence gates

## Frozen baseline ladder

| Identity | Weights | Allowed changes | Current evidence gate |
|---|---|---|---|
| B0 | Official released general_model | Explicit environment compatibility only; original layout/scores/crops/clipping retained | Strict key load, GPU smoke, real benchmark pending |
| B1-score | Same official checkpoint | Sigmoid of BCE head before original score mixture only | Synthetic mathematical validation; accuracy unmeasured |
| B2-layout / B2-neg / B2-combined | Separately retrained controlled models | Validated layout and/or negative-pair correction | Training recipe/data not validated; no runs |
| R1 | Frozen corrected B2 parent | Small source-support fusion controller with matched controls | Not implemented; diagnosis first |

Do not silently use the layout correction with the released checkpoint: its batch-one behavior changes too. B2 must compare an original-code retraining control using the same datasets, recipe, seeds, effective batch, precision and compute budget, not only B0's unknown historical training realization. Report score correction independently of every training correction. Geometry thresholds are ablations, not automatic bugs.

## Minimum experiment record

Unique immutable run ID; parent identity; source git SHA + full dirty-tree content identity; pinned upstream SHA; configuration bytes/hash; checkpoint URL/hash/key validation; dataset version, exact frame IDs/hashes and sequence split; seeds; effective batch/accumulation; optimizer, LR schedule, steps, augmentation, precision; hardware/software; timing scope/warmup/synchronization; peak allocated vs reserved CUDA memory; metric protocol; failure cases; interpretation/limitations. Store failed runs too. Resuming full training will require optimizer/scheduler/RNG state; initialization does not yet supply a training launcher.

## Reproduction and evaluation cautions

- Published values, measurements, correctness tests, and planned work must be labeled separately.
- No new benchmark numbers exist until licensed real sequences are imported and protocols checked. Random weights and synthetic inputs never count.
- Include submitted first-frame prediction when matching upstream OPE. The official rectangle initialization conversion shifts x/y by half a pixel; do not replace it with exact GT silently.
- UAV123, its 10fps derivative and UAV20L can share source sequences. They are distinct protocols, not three independent holdout populations.
- Original training data includes VID, YouTube-BB, COCO, GOT-10k. Audit identity overlap and pretrained data before research claims; official checkpoint training provenance is incomplete.
- Success AUC in the bundled toolkit is the mean of 21 strict-threshold success rates, not a continuous integral. Document this distinction and normalized precision applicability.
- Timing: synchronize GPU; report crop/postprocess and image I/O inclusion explicitly. MACs are not FLOPs and neither establishes latency. Count/profiling is pending rather than inventing an estimate.

## R1 gate

First run a no-training diagnostic: sequence-stratified low-vs-high original pixel support with comparable resized target footprint. Account for crop miss/padding, aspect ratio, speed, occlusion, blur, sequence identity and scale. Use GT only to measure/evaluate diagnostic covariates; any inference controller uses previous predictions and known crop geometry. Source support is a proxy, not proof of image information. Association alone does not establish causal benefit or novelty.

Before substantive R1 training: validated B0/B1 inference and evaluator on actual data; controlled B2 training; meaningful conditional source-support effect with sequence-held-out validation; primary literature matrix; frozen controls (existing fusion, fixed weights, ordinary scale conditioning, capacity-matched controller); preregistered splits/seeds/uncertainty analysis. Reject/revise if support adds no conditional predictive value. R2/R3 remain deferred.
