# Inference scores and geometry audit

Source: immutable HiFT 992aa7c1605360d607af84cb804b93cd8ee81caf. Evidence type below is mathematical/source-level, not tracking improvement.

## Confirmed: BCE logits treated as probability-like scores

`pysot/models/model_builder.py:24,114` trains cls2 with BCEWithLogitsLoss. `pysot/models/utile/utile.py:117-120` returns the raw convolution output without sigmoid. `pysot/tracker/hift_tracker.py:133-135` mixes softmax(cls1)*w2 with raw cls2*w3 and divides by two. With released config w2=w3=1, p=.5 and z=-10 yields -4.75; z=10 yields 5.25. At lines 154-164 that score influences selection and the interpolation learning rate, so scores need not lie in [0,1] and the LR need not be a convex weight.

B1-score uses an isolated output adapter that applies sigmoid(cls2), then calls the original tracker unchanged. For weights 1, the blend is bounded [0,1]. This is a mathematical output-interpretation correction, NOT calibrated probabilities and NOT an accuracy claim. If weights do not sum to 2, the mixture is not normalized automatically. Any retuning would require its own experiment.

## Deliberate heuristics / geometry risks (not automatically defects)

- `_bbox_clip`, lines 65-70, imposes 10px minimum width/height in original-image coordinates. A 2x20px target becomes at least 10x20px when an update is clipped. This can bias tiny/elongated targets; whether it is beneficial regularization is empirical. Preserve in B0/B1-score. On <10px images minimum exceeds boundary, demonstrated by a unit test; those images are outside ordinary UAV benchmark sizes.
- Centers and sizes are independently clipped, so corner coordinates can remain outside the image, e.g. center=(0,0), size=(10,20). This is not full-box containment. Evaluation must not silently clamp corners.
- Original init uses `(w-1)/2` in tracker line 81; official tool first converts rectangle GT through `get_axis_aligned_bbox`, yielding x/y+0.5 at tools/test.py:67-70. Track updates emit `cx-width/2` at 182-185. Half-pixel conventions must be recorded, not repaired without matched evaluation.
- Crop origins are `floor(center-(source_size+1)/2+.5)` in base_tracker.py:51-57. Resizing uses integer rounded crop side but decoder divides by a scale computed before search-size rounding (hift_tracker.py:119-128,162). The actual pixel-center mapping is u=(x-origin+.5)*model/source-.5. It is invertible, but not exactly the ideal unrounded scale for every target size. This is a quantization/design issue pending real localization-bias measurement.
- tanh-like target decoding clips only values <=-1 or >=1 to +/-0.99 (lines 23-26), preventing endpoint log infinities but introducing a discontinuity and not constraining predicted widths/heights positive. Negative/zero decoded extents can destabilize square roots and ratios. No inference fallback is added silently; runner rejects invalid predictions and records failed runs. Need real-checkpoint distributions before design changes.
- Constants grid offset 63 and decode multiplier 143 are tied to the released 287px search configuration and 16px anchor stride. Do not change crop resolution without re-deriving geometry.

## Tests

`tests/test_geometry_scores.py`: deterministic original-vs-sigmoid range tests (including +/-1000 stable logits), OpenCV half-pixel forward/inverse checks for source crop sizes 1/2/7/64/287/1024, padding support, equal resized footprint with different original pixel support, actual upstream clipping behavior. These prove invariants only. Source-support values measure geometry, not true entropy or recoverable visual information.
