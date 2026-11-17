# Wave2 M1: target, identity, and mask correctness

## Decision / G1 acceptance

**Overall G1: PARTIAL. Corrected-API numerical/identity synthetic tests: PASS. Do not declare full M1, the original production dataset/trainer, or a real-data B2 training recipe validated.** The original production dataset has not been migrated to the identity-safe API, and ordinary finite edge-weight=64 behavior is deliberately retained. B0/B1 released-checkpoint inference is untouched. Upstream remains pinned at `992aa7c1605360d607af84cb804b93cd8ee81caf` and unmodified. Wave2 started from project HEAD `5ce4a432aaa3027c7a45d4c678ae04e27a20223a`.

| Acceptance item | Measured result | Boundary |
|---|---|---|
| Ordinary finite positive targets | Bit-identical arrays **and subsequent RNG values**, six bbox geometries × three seeds | Valid nonempty radial support; not a retuning of legacy finite heuristics |
| Empty/collapsed positive support | Actual upstream crash reproduced, corrected finite minimum support, decoded bbox verified | Six cases including tiny, partial crop, edge-clipped non-tiny box; fallback is a chosen B2 heuristic |
| Negative identity safety | Actual source can select identical object; canonical new API rejects equal/missing/malformed identities before constructing labels | Supplied `(dataset,video,track)` metadata must be canonical; cross-dataset duplicates not inferred |
| Empty/singleton/ignored cls1 | Connected finite zero for empty or ignored nonfinite values; singleton has correct NLL/gradient | Invalid active true-class values and invalid labels fail closed |
| IoU/BCE masks | Inactive values selected out before arithmetic; all-zero/empty masks produce connected zero; ordinary finite IoU loss/gradients preserved | Active invalid inputs/reduction overflow raise; no claim that arbitrary upstream NaNs are repaired |
| Composition | 122 CPU tests passed across Wave1 source audit, Wave2 correctness, and the new training graph tests | Synthetic random initialization; no dataset, long training, accuracy, or reproduction claim |
| Patch isolation | All six patches jointly pass applicability check; upstream status empty | Patches not applied; patches alone do not migrate the real-data sampler |

**M1 remains scoped rather than universally closed:** real-data identity/provenance integration, augmented-box frequency statistics, checkpoint/optimizer/schedule/resume fidelity, and the training recipe are not established by these tests. Any gate whose definition requires those is still pending. This report does not substitute looser assertions for failed invariants.

## 1. Positive target crash: verified cause and chosen repair

The actual [target generator](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L47-L66>) clips bbox endpoints to integer grid indices. When both endpoints collapse onto one cell, `w=h=1`, `w//2+h//2=0`, and strict radial `<0` support is empty. Its `se.min()/max()` reduction raises; the second classifier has the analogous reduction at [lines 81–86](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L81-L86>). The failure is not limited to one-pixel physical objects: `(230,230,285,286)` lies within a 287-pixel crop but all endpoints clip to grid index 10, and the same crash occurs.

[CorrectedAnchorTarget](<F:/Code/HiFT/src/hift_research/corrections/positive_targets.py#L47-L155>) mirrors the pinned recipe and makes only declared changes:

1. Reject nonfinite, zero/negative-extent, unrepresentable bbox coordinates, invalid grid configuration, and boxes with no positive overlap with the search crop. **Partially overlapping boxes remain supported**; coordinates are not silently clipped to the crop.
2. If regression radial support is empty, assign weight=1 to the existing clipped center cell; if cls2 radial support is empty, assign target=1 there. This is the **minimum-support policy**, a chosen heuristic rather than a deduction about intended paper behavior. cls1 still follows the existing positive/ignore-band rules, and box offsets still encode the actual bbox.
3. Index the spatial plane explicitly, not `.squeeze()`, so a 1×1 configured map does not become scalar. Singleton support normalization remains finite.
4. Preserve the original exponential tanh expression on finite results, using `np.tanh` only for nonfinite overflow entries. This keeps ordinary labels bit-for-bit while avoiding exponential `inf/inf`. Unrepresentable float32 offset ranges fail closed.

Six fallback regression cases include `(143,143,144,144)`, `(144,144,145,145)`, near-origin and partial-crop objects, the far crop edge, and the clipped 55×56-pixel object. Each test first requires the **unchanged source to raise**, then checks finite corrected arrays, exactly one cls1/cls2/regression support cell, and inverse-encoded box recovery to absolute tolerance `6e-5` pixels. The selected cell can lie **outside** a tiny bbox. For `(144,144,145,145)`, the cell is at `(143,143)` and the left/top offsets are negative; signed offsets are valid, and decoded xyxy correctness is the invariant. No assertion requiring all four offsets positive is introduced.

### Ordinary-positive preservation and retained heuristics

The bitwise controls cover centered, asymmetric, origin-edge, partial-crop, far-edge/nonempty-support, and wide boxes at seeds 0/20261023/31. A separate regression explicitly retains the finite radial-boundary behavior: bbox `(63,63,95,127)` gives maximum regression weight **64**, because [source line 61](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L57-L66>) zeroes only distances strictly greater than the cutoff, leaving raw squared distance on exact boundaries. This is not claimed calibrated or desirable; changing it would violate the requested ordinary-finite preservation and requires a separate controlled recipe decision.

Other retained choices: grid offset 63 / stride 16, integer clipping, radial scale/radii, 1.5 boosted sampled weights, random selection counts, NumPy slice/ignore-band semantics, post-normalization cls2 masking, source search-size//2 encoding scale, and saturation of extreme offsets. Decoding saturated labels is lossy; the tiny-box roundtrip evidence is for nonsaturated cases, not arbitrary huge boxes. No augmented-box occurrence rates or accuracy effect were measured.

## 2. Negative pairs: fail closed on contradictory identities

[Source random-target sampling](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/dataset.py#L128-L137>) does not exclude the template object; [negative branch](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/dataset.py#L233-L241>) can therefore select the same object. A new test executes the actual `SubDataset` class with one video/track/frame: both random targets are necessarily equal. This is a constructive source counterexample, not a measured real-data collision rate.

The new canonical [identity validator and robust pair API](<F:/Code/HiFT/src/hift_research/corrections/positive_targets.py#L16-L44>) require explicit nonempty string triples `(dataset,video,track)`. Frame ID is deliberately excluded: different frames of the same object are still the same identity. Missing/equal/incorrect-arity keys raise `NegativeIdentityError` before label construction. The helper performs no RNG draw, automatic relabeling, or hidden resampling, so it does not silently alter negative prevalence. A caller must explicitly handle a rejected pair under its logged data policy.

Distinct identity-verified negatives retain the Wave1 dense all-background class targets and zero regression weights. Positive pairs delegate to the corrected anchor. The legacy [low-level pair constructor](<F:/Code/HiFT/src/hift_research/corrections/pair_targets.py>) remains for compatibility and is now explicitly documented as **not an identity-verified sampler**; it must not be used directly as the canonical new B2 data entry point. [The new graph's synthetic data path](<F:/Code/HiFT/src/hift_research/training/graph.py#L110-L135>) calls `robust_pair_targets` with explicit synthetic keys.

**Unresolved:** a real-data loader must supply consistent dataset/video/track keys and alias any same-object copies across datasets. The validator checks metadata, not semantic identity from pixels. The original upstream dataset and patches 0001–0006 do not alone supply that adapter. Hence no claim of real-data collision-free sampling.

## 3. Empty, singleton, and numerical mask semantics

### Classification

Wave1's isolated criterion initialized its zero using `pred.sum()*0`: ignored `NaN`/`-Inf` values can poison even an all-ignore result. [The updated criterion](<F:/Code/HiFT/src/hift_research/corrections/classification_loss.py#L13-L33>) uses an empty-slice sum, which is connected to autograd without reading ignored data. It preserves the source 0.5-per-class balancing and ignores -1/-2; singleton indices remain one-dimensional. It validates label cardinality/domain and active true-class values/reductions. Tests inspect both forward values and gradients, including zero gradient at ignored nonfinite rows, not merely `isfinite(loss)`.

This criterion accepts log probabilities, not raw logits. Any NaN must be removed **before** log-softmax if it belongs to an ignored cell; otherwise nonlinear backward can contaminate gradients even with a zero final mask. [Graph-level masked log-softmax](<F:/Code/HiFT/src/hift_research/training/graph.py#L52-L64>) implements that separate boundary.

### IoU and BCE

[Original IoU](<F:/Code/HiFT/third_party/HiFT/pysot/models/loss.py#L33-L65>) evaluates all coordinates and only then multiplies by weight. Actual-source execution with NaN boxes and zero weights returns NaN. [Masked loss adapters](<F:/Code/HiFT/src/hift_research/corrections/masked_losses.py#L13-L84>) instead select positive-weight elements **before** coordinate, IoU, or BCE arithmetic. All-zero and zero-element inputs use differentiable empty sums. Mixed masks give exactly zero inactive leaf gradients, including for inactive NaN/Inf. Weights must be finite/nonnegative; active boxes/logits/targets must be valid; reduction overflow raises explicitly.

Ordinary positive-extent IoU value matches source exactly in the regression, with gradients within rtol=1e-6/atol=1e-7. Unconstrained predicted boxes can have negative extents, so predicted extents are clamped to zero area rather than rejected or changed in decoded coordinates. This is an explicit B2 numerical extension, not a claim that geometry is repaired. Target extents must be positive. The upstream denominator epsilon, added `1e-6` IoU offset, and weighted denominator epsilon are retained: an exact match can have a tiny negative loss. BCE with no mask retains dense mean semantics and soft targets [0,1].

Similarly, ignoring invalid encoded regression only after inverse tanh is insufficient for backward. [Graph-level masked decode](<F:/Code/HiFT/src/hift_research/training/graph.py#L34-L49>) replaces inactive raw encoded values before nonlinear decoding. Criterion-level zero masks do not promise to repair arbitrary NaNs already produced within an upstream model graph.

## 4. Patches and baseline impact

[Patch 0005](<F:/Code/HiFT/patches/0005-empty-positive-support-b2.patch>) implements guarded positive support/validation/encoding in the upstream target class without applying it. [Patch 0006](<F:/Code/HiFT/patches/0006-mask-before-arithmetic-b2.patch>) implements the IoU selected-before-arithmetic path and differentiable empty class loss. The isolated classification validation and identity-aware pair API remain adapter-level policies; the patches do not claim a full real-data trainer. [Patch notes/builders](<F:/Code/HiFT/patches/0005_0006_notes.md>) explain composition. Tests execute patched class bodies in memory and compare them against adapters; no upstream file is overwritten.

All are opt-in B2 changes. Layout correction remains computationally incompatible with treating the official checkpoint as a corrected-training baseline, even though parameter keys/shapes load. No inference B0/B1 adapter, frozen Wave1 report, root setup, or upstream source was edited by this task. No GPU operation, autonomous training job, or benchmark was launched by this worker. Running the supplied training-graph tests does include their bounded synthetic optimizer-step checks; that is correctness evidence, not a trained baseline.

## 5. Verification record

Environment: Python 3.13.0, NumPy 2.4.4, pytest 8.4.2, torch 2.10.0+cu128, Windows AMD64; CPU only for all commands below. NumPy seed=20261023 unless parameterized; Torch source/graph regressions set their own deterministic seeds. These are synthetic test durations, not inference-speed measurements.

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_wave2_correctness.py tests/test_source_audit.py tests/test_training_profile.py -q
```

Measured integrated result: **122 passed, 0 skipped, 10.04 s** (71 Wave2 correctness tests, 36 Wave1 source tests, 15 training-graph tests). The worker ran graph tests supplied by the training worker, but launched no training/profile command or GPU workload. The first Wave2 reference-control run failed because one supposedly ordinary bbox was itself a source crash at the clipped edge. That case was retained as an explicit old-crashes/new-decodes regression, not discarded or given a weaker expected result; an actually nonempty-support far-edge box supplies the preservation control.

Combined `git apply --check` of patches 0001–0006 passed; `git diff --check` passed; upstream `git status --porcelain` remained empty. The full project suite and final commit are lead responsibilities because other workers are editing disjoint scopes concurrently.
