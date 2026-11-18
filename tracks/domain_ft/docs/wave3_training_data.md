# Wave3 real training-input contract (preflight only)

This adapter supports authenticated **COCO train2017** source images and official instance annotations. It is not authority to run SGD, a global identity oracle, a multi-dataset reproduction, or a validated temporal-tracking recipe.

## Interfaces

```python
from hift_research.data_contract.schema import validate_training_manifest
from hift_research.data_contract.pipeline import RealPairPipeline

manifest = validate_training_manifest("artifacts/wave3-training/FROZEN-001/training_manifest.json")
record, arrays = RealPairPipeline("artifacts/wave3-training/FROZEN-001/training_manifest.json").candidate(0)
# arrays=None means explicit rejection; do not draw a substitute.
```

[Validator](<F:/Code/HiFT/src/hift_research/data_contract/schema.py>) verifies source selection/annotation identities, file hashes/dimensions, frozen code/policy hashes, source-only train split, and reserved eval SHA/pHash screens. `status=frozen_preflight_only` and `validation.production_training_authorized=False` are mandatory. Downstream recipe binding must stay provisional.

For a CPU model-input bridge:

```python
from torch.utils.data import DataLoader
from hift_research.data_contract.torch_adapter import CandidateDataset, reject_aware_collate

loader = DataLoader(CandidateDataset(manifest_path), batch_size=4,
                    num_workers=2, multiprocessing_context="spawn",
                    collate_fn=reject_aware_collate)
for batch in loader:
    inputs = batch["model_inputs"]  # CPU FP32, or None if every candidate rejected
    rejected = batch["rejected_records"]
    # Log candidate/rejection counts. No forward/SGD is authorized by this example.
```

[Dataset/collate](<F:/Code/HiFT/src/hift_research/data_contract/torch_adapter.py>) serializes only manifest path/length, constructing pinned dynamic AST classes/caches lazily per process. Pickling after first use and an actual Windows two-worker spawn loader are tested. A scoped global NumPy seed provides **serial per-process** deterministic candidates; it is not thread-safe. Exact six-worker asynchronous loader resume is not claimed. Collation casts accepted source-style arrays explicitly to FP32, separates rejection records, and never fills rejected slots. Original audited array dtypes/hashes remain unchanged.

## Source parity and disclosed deviations

[Source operations](<F:/Code/HiFT/src/hift_research/data_contract/source_ops.py>) execute the pinned preprocessing crop and Augmentation class bodies without upstream modification. A local NumPy proxy supplies only the removed `np.float` alias; global NumPy is unchanged. The `_crop_roi` audit subclass only records affine geometry and calls the unchanged implementation. Tests require identical pixels, boxes, and subsequent RNG draws relative to the original class.

Preprocessing is the actual source SiamFC-style 511×511 mean-padded crop with JPEG quality95 encode/decode, then original centered `_get_bbox`, then template127/search287 augmentation:

- Template shift4, scale0.05, blur0, flip0, color1.
- Search shift64, scale0.18, blur0.2, flip0, color1.
- Gray0; unnormalized BGR→CHW FP32.
- Explicit B2 robust targets preserve source offset63/stride16, normal finite labels, empty-support fallback, dense known negatives, and finite boundary weight64.

The actual preprocessing pixel warp is **510/source_crop_side**, not511/side. Source `_get_bbox` uses the ideal label scale127/context_side; its scale is511/510 times the actual crop pixel scale. This ~0.196% retained mismatch is recorded, not silently repaired. Accepted records include original boxes, centered boxes, augmented boxes for **both** views, preprocessing/augmentation affine matrices, crop JPEG hashes, target shapes/dtypes/domains/support, and final array hashes. Rejected records retain both canonical identities/boxes and reason; augmentation fields are null when rejected before augmentation.

## Identity, candidates, and rejection policy

Canonical identity is `(COCO2017/train2017, source_image_id, official_annotation_id)`, scoped to an annotation instance in one still image. It is **not** an invented global physical-object identity. COCO positives legitimately use the same source annotation/frame000000, matching upstream COCO preparation; they provide no temporal motion/interval evidence.

Each indexed candidate independently draws the **unchanged Pneg=0.05 prior**, then uniform active image→uniform source instance. This bounded COCO-only per-seed policy differs from upstream shuffled global picks and four-dataset mixing/repetition; that deviation is frozen and reported. Rejections do not alter future draws and are never replaced/relabelled:

- Same annotation negative: reject collision.
- Different-image negative: quarantine unknown physical-object alias.
- Same-image distinct instances: allow only if source bbox IoU<0.9; otherwise quarantine ambiguity.
- Crowd/malformed/repeated annotations, duplicate image content, reserved eval exact/near content, missing/changed source files, invalid target shapes/domains, empty positive supervision: reject/quarantine explicitly.

Consequently accepted-negative prevalence is **not forced to5%** and can be far lower. Candidate and accepted fractions/counts/Wilson intervals are reported separately. The accepted subset is not a secretly corrected training distribution.

## Separation and legal limitations

The frozen reservations cover all15 known Wave2/Wave3 UAV source-folder lineages and their known variants, plus2937 acquired eval frame SHA256/pHash64 values. Training rejects UAV123/UAV20L/UAV123@10fps/DTB70 as source datasets. SHA equality and pHash Hamming≤4 screen known exact/near duplicates; neither certifies absence of semantic, cropped, transformed, or unseen cross-dataset aliases. Those remain unresolved, and unknown cross-image negatives are quarantined. Eval GT boxes/flags are never used as training labels.

Official image/annotation origins, original member CRC/hash, image license IDs/URLs, and COCO license metadata are retained. Source terms are recorded, not a legal opinion or blanket grant for all photographs. VID/GOT/YTBB are not registered in this bounded preflight: their assets/provenance are absent; no guessed download or benchmark substitution is used.

## Artifacts and replay

[Acquisition](<F:/Code/HiFT/artifacts/wave3-training/acquisition-001/acquisition.json>) tracks all training network bytes under the approved192MiB cap and≤32 images. [Frozen contract](<F:/Code/HiFT/artifacts/wave3-training/FROZEN-001/training_manifest.json>) precedes all1024 preflight candidates. [Preflight](<F:/Code/HiFT/artifacts/wave3-training/PREFLIGHT-001/>) stores accepted compressed NumPy arrays, every candidate/reason, immutable array/file hashes, statistics, and independent fresh-pipeline replay checks. No model, GPU, or SGD is used.
