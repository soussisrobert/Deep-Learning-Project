# Source-resolution-aware hierarchical fusion: test before building

**Status: hypothesis and preregistration draft, 2026-10-23. No implementation, training, or measured outcome.** Bibliographic evidence and limitations are in the [primary-source review](../reports/literature_review.md). Priority is baseline correctness and falsification, not a novelty assertion.

## 1. Define the variable before proposing a gate

Let a target have original-frame dimensions `(w, h)` pixels; a square source search crop has side `L` pixels and is resized to network side `M`. Define:

- **Geometric source support:** `(w, h)`, optionally area `w*h` and minimum side `min(w,h)`. This is not true optical resolution or a count of independent informative samples.
- **Resized footprint:** `(p_w, p_h) = (M*w/L, M*h/L)` network pixels. Approximate feature/token occupancy at stride `d` is `(p_w/d, p_h/d)`, not `(w/d, h/d)`.
- **Sampling scale:** `a = M/L`; padding/containment, aspect ratio and visibility are separate variables. Upsampling can enlarge footprint but creates no new acquired detail.

In pinned HiFT, when the previous predicted box matches these target dimensions, and ignoring rounding/boundaries and its large-box fallback,

`S = sqrt((w + c*(w+h)) * (h + c*(w+h)))`, `L = k*S`,

where `c` is the context amount and `k` the instance/exemplar size ratio. Scaling both target dimensions by `r` scales `S` and `L` by `r`, so **resized footprint is unchanged** at fixed aspect ratio. Thus a 10-pixel and 100-pixel source target can have similar network footprint but different available detail. Conversely, changing the search crop changes footprint/context without changing acquired target detail. Predicted box error, rounding, padding, scale lag and the fallback break exact cancellation.

Evidence: [crop sizing](../third_party/HiFT/pysot/tracker/hift_tracker.py#L117-L128), [source crop and resize](../third_party/HiFT/pysot/tracker/base_tracker.py#L38-L95), upstream SHA `992aa7c1605360d607af84cb804b93cd8ee81caf` (read-only inspection). A future inference gate could use **past predicted** source dimensions, crop transform and image/response quality; ground-truth dimensions are permitted only for offline diagnosis, never as a deployed controller input after initialization.

## 2. Narrow hypothesis and alternatives

**H1:** On a fixed, validated HiFT baseline, the relative usefulness of hierarchical cues varies with available source sampling detail after controlling resized footprint and pre-existing nuisance variables. A source-support cue might therefore add value beyond a footprint-only or quality-only policy. The direction is unknown: shallow texture may help localization when reliable, but become noise when poorly sampled. Do not assume “smaller source target ⇒ more shallow weight.”

**H0 / competing explanations:** ordinary footprint/stride occupancy, crop containment, aspect changes, motion, occlusion, blur, confidence miscalibration, sequence identity, or template mismatch explain the apparent effect; existing content attention may already provide all useful adaptation. More source pixels need not imply more information. A low-resolution attribute score alone cannot distinguish these explanations.

Closest verified comparisons: HiFT/SiamHFFT/SiamHSFT (hierarchical fusion), AMST² (temporal fusion), LoReTrack (input-resolution distillation), ZoomTrack (sampling allocation), SAMViTrack (motion-dependent crop), TSDTrack/ATOM (quality), GTUTrack (RGB–thermal template quality). None of their existence or absence establishes novelty for H1.

## 3. Cheapest falsification diagnostic: R1-A, no training

**Execution gate:** proposal only. First validate geometry on CPU and the actual baseline/checkpoint. B0 is released behavior + official checkpoint; B1 independently validated inference corrections; B2 corrected training with controlled retraining; R1 is the later candidate on frozen B2. Never rename B0/B1 to B2. If only B0 is available, any diagnostic is explicitly exploratory B0 evidence and cannot authorize R1 training. No random/partially loaded checkpoint, benchmark download, new head, or learned gate is needed here.

### A. Geometry/metadata screen (CPU, cheapest stop)

From existing permitted diagnostic sequences, log source and predicted box dimensions, crop transform/rounding, footprint, padding, target-in-crop fraction, normalized target displacement, aspect ratio/change, visibility/occlusion and blur proxies. Ground truth supplies **offline labels only**. Inspect source-support versus footprint jointly, not isolated size bins. If nominal support groups have no within-sequence overlap in footprint/motion/visibility, the dataset cannot identify the proposed relation; do not fit across disjoint groups and call it evidence. Missing annotation is “unknown,” not “unoccluded.”

### B. Same-pair sampling intervention (small frozen forward-pass probe)

Use, for example, **12 prespecified sequence families × 8 spaced frame pairs**, selected without seeing model outcomes; reserve four families for confirmation. This is a budget cap, not a power guarantee. Exclude target truncation/padding and use visible, contained targets for the primary probe; retain failures separately as a containment diagnostic. Fix the template, source crop boundaries, target location, resized footprint, aspect ratio, frame time and baseline tracker state across each paired test.

1. Cache an unmodified source crop. Generate antialiased downsample–upsample versions at fixed ratios `q ∈ {1, 1/2, 1/4}` **before the identical final resize**. The effective sampling grid is reduced while the target's geometric box and final footprint remain fixed. Use one documented interpolation/filter and a sham identity pipeline; repeat on a second filter only if the primary result survives. This models lost sampling detail, not a newly acquired smaller object or every real camera degradation.
2. Freeze all model parameters, evaluation mode, preprocessing, decoder and postprocessing. Replay every condition from the **same cached state**, without propagating predictions or updating the template. Search-only degradation is primary; degrading template and search together is a later mismatch control. A one-step result is not a closed-loop benchmark gain.
3. At existing projected correlation maps, before positional addition, apply prespecified **small symmetric** shallow-versus-deep gain perturbations (e.g. `ε = 0.1`: gains `(1+ε, 1, 1−ε)` and its reverse), alongside unmodified fusion. [HiFT fusion inputs](../third_party/HiFT/pysot/models/utile/utile.py#L87-L110) feed a shared transformer/head: there are **no independently trained per-level predictors** here. Scale perturbations test local sensitivity only; hard layer deletion is an out-of-distribution intervention and cannot establish an optimal fusion policy. If gain magnitude changes conclusions, stop rather than tune it on confirmation data.
4. Primary measurement is the **within-pair change in shallow-versus-deep perturbation contrast** as `q` decreases, using one-step IoU in source coordinates. Record normalized center error and raw response statistics secondarily; avoid fixed-pixel precision as the sole outcome because it favors large targets. Mere degradation of accuracy with lower `q` does not support fusion adaptation.

For IoU `Y(q,g)`, prespecify the interaction contrast:

`I(q) = [Y(q,g_shallow) − Y(q,g_deep)] − [Y(1,g_shallow) − Y(1,g_deep)]`.

Use both signs as an exploratory possibility on development sequences, then lock the direction for held-out confirmation. The probe changes feature magnitude and can shift attention; it does not isolate a pure semantic “level contribution.” No oracle-selected per-frame perturbation is a deployable policy.

### C. Necessary controls and inference limits

| Potential false explanation | Minimum control |
|---|---|
| Crop containment / extra context | Identical crop in sampling intervention; offline verify complete containment and border margin. A separate fixed-input crop-size sweep tests footprint/context and is not mixed into the main contrast. |
| Aspect / motion / occlusion | Same frame pair fixes these under intervention. For natural-support analysis, compare within sequence and matched aspect/change, displacement normalized by target size, visibility and blur strata. Report excluded/unknown cases and fast-motion/occluded groups separately. |
| Quality rather than source support | Include same-geometry blur degradation and report pre-treatment quality strata; compare support patterns at overlapping footprint and quality. Confidence/entropy are not calibrated resolution estimates. Post-treatment confidence is a mediator: report it, but do not regress it away in the primary total-effect contrast. |
| Background degradation / template mismatch | Whole-crop resampling also changes background; disclose this. Confirm with blur and template/search degradation controls before attributing the effect specifically to target detail. Avoid claiming a causal original-target-size effect from this intervention alone. |
| Sequence / dataset confounding | Equal weight per sequence family; paired differences, sequence-cluster bootstrap intervals and leave-one-family-out sensitivity. No frame-random split. UAV123, UAV123@10fps and UAV20L overlap: do not treat their derived sequences as independent replications. Audit training/pretraining overlap. |

**Preregistered decision draft:** fix a smallest useful interaction (for example **0.01 absolute IoU**, to be agreed before running), metric, signs, exclusions and family split. If the held-out interval lies entirely inside `[-0.01, +0.01]`, reject this local fusion-sensitivity premise at that tolerance. A wide interval is **inconclusive**, not equivalence. If effects disappear after nuisance matching, reverse across sequence families, track only the footprint sweep/blur, or require oracle source boxes, do not proceed with the source-support gate rationale. A surviving interaction motivates only a further controlled diagnostic; it neither proves learnability/generalization nor establishes novelty. A null local probe does not rule out every possible retrained architecture.

## 4. Evidence required before any candidate

No experiment is authorized or implemented by this document. Preserve immutable manifests with root revision/dirty-diff hash, upstream SHA, config/checkpoint hashes and strict load status, data/split and overlap audit, frame selection, seeds, software/hardware, filters/crop transforms, frozen-state policy, all exclusions, metrics/runtime and uncertainty. A later candidate must compare equal-budget frozen baseline, footprint-only, quality-only and source-plus-footprint conditions on sequence-disjoint data; gains from B2 retraining must never be credited to R1. Stop at unresolved baseline/assets rather than fabricate results.
