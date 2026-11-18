# CPU evaluation and asset/protocol audit

## Status and evidence boundaries

**Implemented and measured:** deterministic CPU NumPy evaluator, strict local asset import/validation, GT-free inference-input export, immutable JSON reporting, and synthetic fixtures. Latest command:

```text
PYTHONPATH=src python -m pytest tests/test_evaluation.py
Python 3.13.0 / NumPy 2.4.4 / pytest 8.4.2 / Windows
35 passed in 0.62s
```

Direct checkout CLI `--help` was also verified for both scripts without an editable install or `PYTHONPATH`. No GPU, training, dataset download, or benchmark inference was performed in this task. **There are no measured benchmark scores here.** Synthetic perfect-box `20/21` and boundary examples are metric fixtures, not model performance. Artifact [schema](../experiments/schema.json) was JSON-parsed and checked for expected artifact variants by fixtures; a full external Draft 2020-12 schema-validator run is not claimed.

**Local availability, independently hashed within the project:** full [general_model.pth](../artifacts/checkpoints/general_model.pth), 86,039,991 bytes, SHA-256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`; upstream [alexnet-bn.pth](../third_party/HiFT/pretrained_models/alexnet-bn.pth), 15,013,823 bytes, SHA-256 `933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8`. The lead's [B0 GPU smoke manifest](../artifacts/runs/INIT-GPU-B0-002/manifest.json#L65-L123), independently read for this audit, records official weights strictly loaded, finite output tensors, 11,066,312 parameters, RTX 5070 Ti / torch 2.10.0+cu128, and synthetic warmed forward timing only. This clears strict-load/synthetic-execution preparation, not benchmark or original-environment equivalence. Hashing alone would not certify those gates. No benchmark image/annotation collections or released result archive were found in this project. No search of unrelated host directories was performed.

## Pinned source inspection

Upstream submodule commit: `992aa7c1605360d607af84cb804b93cd8ee81caf`. Inspected without modifying it:

- [Statistics](../third_party/HiFT/toolkit/utils/statistics.py#L6-L48): continuous rectangle IoU; strict `>` success thresholds from `np.arange(0,1.05,0.05)`; center-error comparison `<=`.
- [OPE benchmark](../third_party/HiFT/toolkit/evaluation/ope_benchmark.py#L16-L21), [evaluation methods](../third_party/HiFT/toolkit/evaluation/ope_benchmark.py#L23-L122), [aggregate reporting](../third_party/HiFT/toolkit/evaluation/ope_benchmark.py#L124-L160): `(w-1)/2` center convention, all GT rows as denominator, sequence-macro averaging, normalized precision reporting at curve index 20 (threshold 0.20), not normalized AUC.
- [UAV123 loader](../third_party/HiFT/toolkit/datasets/uav.py#L34-L50): explicit JSON sequence/frame lists required.
- [10fps loader](../third_party/HiFT/toolkit/datasets/uav10fps.py#L12-L61), [UAV20L loader](../third_party/HiFT/toolkit/datasets/uav20l.py#L13-L63), [DTB70 loader](../third_party/HiFT/toolkit/datasets/dtb.py#L75-L127): hardcoded current-directory paths; sorting and row/image count issues are printed rather than treated as a failed import. Raw 10fps/20L metadata paths omit the `data_seq` component, making dataset-root alignment another explicit integration concern.
- [Prediction loader](../third_party/HiFT/toolkit/datasets/video.py#L31-L56): missing/count-mismatched results do not produce a clear fail-closed validation contract.
- [Upstream evaluation CLI](../third_party/HiFT/tools/eval.py#L18-L42): parser declares `dataset_dir`, `tracker_result_dir`, and `trackers`, but later references undeclared `tracker_path`/`tracker_prefix`. Multiple dataset aliases/capitalizations disagree with [factory dispatch](../third_party/HiFT/toolkit/datasets/__init__.py#L18-L27). This new standalone evaluator avoids those entrypoint defects; it does not patch upstream.
- [Upstream inference initialization](../third_party/HiFT/tools/test.py#L64-L79) and [rectangle conversion](../third_party/HiFT/pysot/utils/bbox.py#L127-L134): first submitted rectangle is shifted by `+0.5` in x/y relative to raw GT. That submitted initialization frame is included by OPE; replacing it with GT would silently change B0 results.

## Implemented metric contract

[Metrics implementation](../src/hift_research/evaluation/metrics.py) uses CPU NumPy only; no tracker, PyTorch, image GT controller, network, or GPU dependency.

1. Success IoU areas use `x+w,y+h`, no extra pixel in area computation. Evaluate strict `IoU > threshold` at the exact upstream 21-point floating grid. “AUC” is the **arithmetic mean**, not trapezoidal integration or mean IoU. Perfect positive identical boxes yield `20/21`; exact IoU 0.5 yields `10/21`.
2. Pixel precision is the fraction of expected rows whose Euclidean center error is `<=20`. Centers use `x+(w-1)/2,y+(h-1)/2`. Zero/negative coordinates do not make an otherwise valid rectangle invalid.
3. Optional normalized precision uses `sqrt((dx/gt_width)^2+(dy/gt_height)^2)`, with point report `<=0.20` and curve 0 through 0.50. It is labeled **supplemental_diagnostic** because pinned benchmark CLI never requests it. Default is null/not requested, not a misleading zero. Positive GT dimensions are required; normalization never enters inference.
4. Compute each sequence's curve over its full expected row count, then average sequences equally. Initialization is included exactly as submitted. No restarts, dropped rows, frame pooling, hidden warm-up exclusion, or GT substitution.
5. Default `error` rejects invalid GT/predictions. Explicit `count_as_failure` keeps invalid rows in all denominators and assigns failures for every threshold. Missing/extra rows always fail, even under that policy. Counts and policy are included in reports.

## Deliberate deviations and limitations

- Pinned success masks nonpositive GT dimensions while retaining the full frame denominator; pinned center precision masks nonpositive GT centers, initializes masked distances to `-1`, and can consequently **count invalid/masked centers as successes**. This implementation does not preserve that precision defect. Valid negative-center boxes are scored geometrically; malformed GT is rejected or explicitly failed. Cross-tool equivalence is asserted only for valid positive-center fixtures, not every malformed source annotation.
- Pinned normalized implementation adds `1e-16` to GT dimensions. This implementation requires positive dimensions and uses direct center displacement divided by those dimensions, avoiding a hidden zero-dimension pseudo-definition. Exact numerical boundary behavior outside ordinary scales is not promised identical to the legacy epsilon-based arithmetic.
- No benchmark-specific absent/occluded visibility mask, VOT restart/status format, polygon conversion, OTB special filename fallback, or confidence-based long-term protocol is inferred. Such inputs fail or require an explicitly documented adapter/protocol extension. Do not silently delete NaN annotations or source absence rows to make a run pass.
- Numeric overflow and degenerate area underflow fail validation; they do not yield NaN/Infinity report metrics. Source box positions are not clipped to the image because clipping changes the metric protocol.
- Dataset imports enforce explicit names, duplicate/case-collision safety, frame/GT lengths, first GT initialization equality/validity, root-contained paths, nonempty frames, annotation hashes, stored image sizes, and optional image hashes. They do **not** certify official sequence membership/ranges, count-preserving pre-import image substitution, decoding, or temporal sampling correctness. `expected_count_only` never means verified official membership.
- Source-image full hashes are opt-in because reading an entire local benchmark can be expensive. Without them, same-sized image changes are undetectable. An imported hashed manifest revalidates image contents on evaluation; all result predictions and the dataset manifest are always hashed.
- Git tracked-diff hashes exclude untracked content, disclosed in output. Evaluator files are independently hashed; checkpoint/config/frame provenance must additionally be preserved in the linked lead inference manifest. A valid evaluation report certifies how boxes were scored, not how they were produced.

## Protocol sources and exact outstanding assets

The official [UAV resource page](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset) specifies UAV123's 123 sequences, upright boxes, overlapping long-term UAV20L content, a separate downsampled UAV123_10fps archive, annotation-details document, and modified OTB configuration files. Its listed large archives were not downloaded. The official [DTB70 project](https://github.com/flyers/drone-tracking) states its dataset format follows OTB50; its [sequence configuration](https://raw.githubusercontent.com/flyers/drone-tracking/master/experiments/util/configDTBSeqs.m) gives exact names and start/end ranges.

Required before a full benchmark claim: official four-variant images/annotations; original archive hashes; official sequence/segment configurations; annotation coordinate-origin and absence semantics; variant-specific sampling phase; all-image decode preflight; source-video/split mapping; strictly validated full checkpoint and configuration identity; and immutable lead inference manifests. UAV123 raw shared source folders must not be mistaken for the 123 official evaluation segments. The adapter requires exact frame lists through toolkit-compatible metadata rather than inventing segment boundaries. Full collection counts (123/123/20/70) are enforced unless the user explicitly opts into a labeled partial import.

## Leakage controls

- Only offline evaluation sees per-frame GT. [Asset importer](../scripts/import_assets.py) can export GT-free sequence inputs with frames plus initialization, after an explicit coordinate-origin declaration. It preserves the upstream rectangle half-pixel initialization convention. Exported input contains no later GT; inference must enforce that boundary independently.
- [Evaluation CLI](../scripts/evaluate.py) consumes predictions after inference; submitted initialization is never overwritten. It can explicitly align zero-origin predictions with annotation-origin GT. A one-sequence run is marked partial. Result directories with extra sequence files fail rather than silently subset.
- UAV123, UAV123@10fps, and UAV20L share content. Do not describe them as independent evidence or split by frame/variant to create false disjointness. Group by original source video/identity before training/validation/test division.
- Freeze tracker/config choices before benchmark tests. Subsequent failure analysis is diagnostic, not a fresh independent test set. Do not tune on benchmark GT, choose easy sequences, stop on GT-derived failures, or feed GT to crop/context policies after initialization.
- Released checkpoint training membership/recipe is not established merely by upstream dataset-preparation instructions. Verify VID/YouTube-BB/COCO/GOT-10k provenance and overlap where feasible; unresolved membership is an explicit leakage risk. No retraining occurred here.

## Fixture coverage and reproducibility

[Tests](../tests/test_evaluation.py) cover perfect/disjoint boxes; exact IoU/20px/normalized boundaries; unequal-length sequence macro weighting; zero/negative centers; initialization inclusion; invalid/NaN/infinite/zero/negative-dimension boxes; missing/extra rows; arithmetic extremes; pinned upstream differential comparison on seeded valid positive-center boxes; malformed/empty CSV rows including missing comma fields; partial-vs-complete imports; duplicate/path-traversal/missing frames; image and annotation tampering; raw 10fps/20L numeric frame ordering without resampling; UAV123 metadata/initialization requirements; nonfinite annotation serialization under explicit failure policy; hash-pinned checkpoint copying without unpickling; no-overwrite manifests; duplicate JSON keys; origin-declared GT-free export; CLI end-to-end scoring; and unexpected prediction files.

Usage and asset preparation commands are in [dataset procedures](../docs/datasets.md). Artifact schema is [schema.json](../experiments/schema.json). No out-of-scope root configuration, upstream, literature, correction, or inference code was changed by this worker.
