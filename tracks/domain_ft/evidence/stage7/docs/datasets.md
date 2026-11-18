# Dataset, checkpoint, and evaluation procedure

## Scope and evidence

This procedure evaluates **precomputed boxes on CPU with NumPy**. It does not train, download datasets, run a GPU, or certify paper reproduction. Synthetic fixtures prove mathematical/interface behavior, not tracking accuracy. Upstream is immutable at `992aa7c1605360d607af84cb804b93cd8ee81caf`.

Canonical dataset names are `UAV123`, `UAV123@10fps`, `UAV20L`, and `DTB70`. Upstream aliases are inconsistent (`UAV10fps`, `UAV20l`, `UAV20` in different places); do not use aliases as separate datasets.

## Exact assets required and project-local availability

Inventory is restricted to this project, not the rest of the host. At this implementation's audit:

- Released HiFT full checkpoint: [general_model.pth](../artifacts/checkpoints/general_model.pth), **86,039,991 bytes**, SHA-256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`. The lead acquired the exact Drive object linked by the pinned upstream [README](../third_party/HiFT/README.md#L22-L25), object ID `1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA`. Local hash was independently recomputed. **Hash integrity is verified. The lead's [GPU smoke manifest](../artifacts/runs/INIT-GPU-B0-002/manifest.json) records strictly loaded official weights, finite synthetic outputs, and 11,066,312 parameters on RTX 5070 Ti / torch 2.10.0+cu128. This is execution evidence, not benchmark reproduction or original-environment numerical equivalence.** Do not substitute a backbone for the full tracker checkpoint.
- Upstream pretrained backbone: [alexnet-bn.pth](../third_party/HiFT/pretrained_models/alexnet-bn.pth), **15,013,823 bytes**, SHA-256 `933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8`. Available, but not a replacement full HiFT checkpoint.
- All four benchmark image/annotation collections: **not available in the project at audit**. No benchmark scores are reported.
- Official sequence configuration, start/end ranges, annotation convention document, archive hashes, and exact local derived frame manifests: **not yet verified/imported**. Count checks alone are not official membership checks.
- Optional released HiFT result archive for cross-checking evaluation: not imported. The pinned [README](../third_party/HiFT/README.md#L56-L63) links Drive object `1OjwuX1i-UxzzMxngrDIc-OqY9e9iV2nI`. It is not required to score newly generated results, and must not be mistaken for locally measured predictions.

Sources inspected: official [UAV dataset page](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset), official [DTB70 repository](https://github.com/flyers/drone-tracking), [DTB70 sequence configuration](https://raw.githubusercontent.com/flyers/drone-tracking/master/experiments/util/configDTBSeqs.m), and pinned local loaders/metrics. External links were inspected as documentation only; no archives were downloaded by this evaluator task.

## Protocol identity and required layout

All four use one-pass evaluation (OPE): initialize once on the first frame, track forward, no GT-driven restarts, and retain every expected frame. Evaluate upright `x,y,width,height` rectangles. Preserve official temporal ordering and sequence boundaries.

- **UAV123:** 123 official annotated sequence segments. The official page reports more than 110K frames and provides the full UAV123/UAV20L archive (~13.7GB), annotation details, and modified OTB configurations. A raw directory count is not enough: shared source videos can contain multiple named evaluation segments with different start/end ranges. Import requires toolkit-compatible metadata named `UAV123.json` at the supplied root. Each sequence must provide `img_names`, `gt_rect`, and `init_rect` (equal to its first GT row). Paths are root-relative. Obtain/derive this metadata from the **official sequence configuration**, recording its original hash; never guess boundaries by sorting shared source folders.
- **UAV123@10fps:** 123 segments from the official separately downsampled collection (~4.6GB). Required prepared layout: `ROOT/data_seq/SEQUENCE/FRAME.jpg` and `ROOT/anno/SEQUENCE.txt`. This importer does **not** downsample UAV123, reconstruct official frame phases, or assume that taking every third arbitrary frame reproduces the official variant. Preserve official segment ranges and variant-specific annotations. Original image IDs need not be contiguous after downsampling.
- **UAV20L:** 20 long-term source sequences, overlapping the UAV123 family. Required prepared layout is also `ROOT/data_seq/SEQUENCE/FRAME.jpg` plus `ROOT/anno/SEQUENCE.txt`. Use the official long-sequence annotations/ranges; do not concatenate guessed UAV123 segments. “Long-term” does not imply use of VOT-LT presence/F-score protocol here: the pinned toolkit evaluates rectangle OPE.
- **DTB70:** 70 drone sequences in OTB-style layout `ROOT/SEQUENCE/img/FRAME.jpg` plus `ROOT/SEQUENCE/groundtruth_rect.txt`. The official configuration provides exact names and frame endpoints. Compare the imported manifest against that configuration before claiming complete-benchmark results.

The raw-layout importer sorts image filenames numerically, enforces image/annotation counts and sequence-name correspondence, and records exact paths. It does not certify official names/ranges, detect a count-preserving replacement of an image before import, decode image contents, or repair source annotations. Use `--hash-frames` for byte-level identity, then separately confirm all frames decode using the inference preflight. Without it, identity is explicitly `paths_and_sizes_only`; same-sized changed image contents cannot be detected. Never label count-only validation “official complete dataset.”

## Import local assets (no network)

From the project root; either an editable install or the direct-script checkout bootstrap works:

```bash
python scripts/import_assets.py hash artifacts/checkpoints/general_model.pth
python scripts/import_assets.py checkpoint --source /explicit/local/general_model.pth --destination artifacts/checkpoints/general_model-imported.pth --sha256 8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101 --output artifacts/manifests/checkpoint-import-001.json
python scripts/import_assets.py dataset --dataset DTB70 --root datasets/DTB70 --output artifacts/manifests/dtb70-001.json --hash-frames
python scripts/import_assets.py validate-dataset artifacts/manifests/dtb70-001.json
```

Paths in examples denote assets the user must explicitly supply; they are **not present just because the command is shown**. Checkpoint import requires a trusted expected hash and exclusively creates a new destination. It does not deserialize pickle, validate model keys, or establish publisher authenticity merely because an arbitrary supplied hash matches. The lead inference loader must use safe loading and strict architecture key/shape checks before any benchmark claim.

All manifest/result writes refuse to overwrite existing files. `--allow-partial` is only for explicitly labeled fixtures/subsets, never a way to silently ignore a missing sequence in a reported benchmark. Default invalid-box policy is `error`. If source annotations contain absent/invalid frames, inspect the official annotation convention first; this implementation does not assume a benchmark-specific visibility mask. An explicitly chosen `--invalid-policy count_as_failure` preserves such rows as failures and discloses the deviation. It still requires a valid first-frame initialization.

## Separate evaluation GT from inference inputs

Full dataset manifests contain all GT **for offline evaluation only**. Never pass them to a tracker/controller. Use the exporter to create separate per-sequence inputs containing only ordered frames and a tracker-ready initialization box:

```bash
python scripts/import_assets.py dataset --dataset UAV123 --root datasets/UAV123 --output artifacts/manifests/uav123-002.json --coordinate-origin 1 --inference-dir artifacts/inference_inputs/uav123-002 --hash-frames
```

`--coordinate-origin 1` above is an **example, not a verified claim about every archive**. Explicitly inspect the official annotation convention; choose `0` or `1` accordingly. No origin is guessed. Export subtracts the declared origin from `x,y`, leaves widths/heights unchanged, then preserves the pinned upstream rectangle initialization convention: `x += 0.5`, `y += 0.5`. Upstream [get_axis_aligned_bbox](../third_party/HiFT/pysot/utils/bbox.py#L127-L134) gives `cx=x+w/2`; upstream [test initialization](../third_party/HiFT/tools/test.py#L64-L79) converts back with `cx-(w-1)/2`, hence the half-pixel shift. Removing it belongs to a separately identified correction baseline, not an unannounced B0 change.

Exported JSON includes `name`, `dataset`, absolute `frames`, `initial_bbox`, `coordinate_origin:0`, provenance fields, `initialization_convention:upstream_tools_test_rectangle_half_pixel`, and `evaluation_ground_truth_included:false`. There are no subsequent GT boxes. Inference must never read evaluation-only GT, including through logging, adaptive crop selection, sequence filtering, early stopping, or parameter tuning.

## Evaluate saved predictions reproducibly

Prediction files have exactly one numeric four-column `xywh` row per expected frame, **including the submitted initialization prediction**; comma or whitespace separated, no header, empty lines, VOT status rows, confidence column, or polygons. Negative/zero positions are valid; width/height must be positive unless the explicit failure policy is selected. A missing row, extra row, missing sequence, duplicate file, unexpected sequence file, or ambiguous `.txt`/`.csv` pair is an error. Rows are never truncated, interpolated, paired with `zip`, or silently dropped.

For a directory containing exactly `SEQUENCE.txt` or `SEQUENCE.csv` per manifest sequence:

```bash
python scripts/evaluate.py --dataset-manifest artifacts/manifests/dtb70-001.json --predictions results/DTB70/B0 --output experiments/runs/dtb70-b0-evaluation-001.json
```

For one lead-runner output, supply the explicit sequence name. Inputs exported above produce **zero-origin** predictions, so use the origin conversion flag to align GT:

```bash
python scripts/evaluate.py --dataset-manifest artifacts/manifests/uav123-002.json --predictions experiments/runs/example/predictions.csv --sequence EXPLICIT_SEQUENCE --prediction-coordinate-origin zero --run-manifest experiments/runs/example/manifest.json --output experiments/runs/example/evaluation-001.json
```

This is labeled **partial**, not a full-benchmark result. For original upstream predictions in unchanged annotation coordinates, use the default `annotation_native`; never shift only one side without declaring it. `--invalid-policy` must match the import manifest exactly. `--normalized-precision` opts into a supplemental diagnostic; omission produces JSON nulls, not fabricated zeros.

Reports record exact dataset-manifest and prediction hashes, evaluator-source hashes, pinned upstream SHA, available git revision/tracked diff hash, untracked-file listing, CPU/Python/NumPy/platform, seed, time, policies, sequence scores, and macro scores. A supplied inference manifest is linked by hash, not automatically trusted. Git diff hashes exclude untracked content, explicitly disclosed; source hashes cover evaluator files. Preserve the matching lead inference manifest for checkpoint/config/source-frame/hardware identity and timing. Evaluation runtime is CPU scoring overhead, **not tracking FPS**. Structural artifact definitions are in [schema.json](../experiments/schema.json); cross-field and local-file checks are implemented in [assets.py](../src/hift_research/evaluation/assets.py).

## Metric convention and deviations

- **Success AUC:** IoU uses continuous rectangle extents `x+w,y+h`, with no extra `+1` in areas. Thresholds are exactly upstream `np.arange(0,1.05,0.05)`, i.e. 21 points from 0 through 1. A frame succeeds iff `IoU > threshold`, **strictly**. A sequence's “AUC” is the arithmetic mean of these 21 success rates, not a trapezoidal integral or mean IoU. Even identical positive boxes score `20/21`, because `IoU > 1` is false. Changing to `>=` or dropping the endpoint changes the protocol.
- **Precision@20px:** center is `(x+(w-1)/2, y+(h-1)/2)` and Euclidean center error succeeds iff `<=20` pixels. Full curve uses thresholds 0 through 50 pixels.
- **Normalized precision:** diagnostic `||((pred_center-gt_center)/gt_wh)||_2`, thresholds 0 through 0.50 by 0.01; reported point is `<=0.20`, **not normalized-precision AUC**. Width and height normalize their own axes using GT dimensions, solely offline. Pinned OPE code supports it, but the pinned evaluation CLI does not call it for these benchmarks. Do not claim it is a primary HiFT paper metric. Nonpositive GT dimensions have no normalized meaning and are rejected or explicitly counted as failures.
- **Aggregation:** first score frames within each sequence, then average sequences equally. A long sequence does not outweigh a short one. Include first-frame predictions exactly as supplied, as the pinned OPE does. No automatic GT substitution or initialization exclusion.
- **Disclosed stricter/corrected behavior:** pinned loaders often print a count mismatch or missing result rather than validate. This wrapper raises. Pinned precision masks GT centers with positive coordinates and initializes masked distances to `-1`, accidentally counting those invalid centers as success; this evaluator does not reproduce that bug. Negative/zero valid centers are scored normally. Nonfinite/nonpositive boxes are rejected by default; `count_as_failure` assigns failed IoU/center tests while retaining every denominator. This can differ from upstream invalid-GT behavior and is recorded, not hidden. Arithmetic overflow/degenerate underflow is rejected, not serialized as a score. Dataset absence masks/restart protocols are not inferred or implemented.

## Leakage precautions and remaining acceptance gates

1. Freeze model/config/algorithm and split identities before benchmark evaluation. Never use per-frame GT after initialization to drive inference. Select failures for diagnosis only after an immutable initial run and keep diagnostic versus final evaluation identities separate.
2. UAV123, its 10fps variant, and UAV20L share source content. They are not independent validation evidence. Partition by original source video/identity, not just segment name or sampled frame; never train/validate/test on overlapping variants.
3. Audit training/pretraining datasets and official checkpoint provenance. Upstream lists VID, YouTube-BB, COCO, and GOT-10k preparation; that list alone does not prove the precise released training recipe or absence of overlap. Treat unknown membership as an unresolved leakage risk. Record deduplication policy and source-video mappings before retraining.
4. Validate official names, segment endpoints, downsampling phase, absent-frame semantics, coordinate origin, frame decoding, archive hashes, and split membership. Count-only import is useful preparation, not a finished benchmark gate.
5. Run deterministic CPU fixtures first:

```bash
PYTHONPATH=src python -m pytest tests/test_evaluation.py
```

No GPU or training is needed. See [evaluation audit](../reports/evaluation_audit.md) for measured synthetic-test evidence and explicit limitations.
