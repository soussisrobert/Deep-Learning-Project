# Wave3: authentic training-input contract and 1024-candidate preflight

## Gate decision and primary result

**G1 PASS, bounded scope:** a frozen, identity-aware real-input contract executed all1024 canonical attempts with source-parity crop/augmentation, checked B2 targets, persisted arrays, and deterministic replay. **G5 PARTIAL / no production training authorization:** all51 negative candidates were quarantined for unknown cross-image physical identity. The accepted cohort has **zero negative pairs**, is COCO still images only, and must not be presented as a valid5%-negative temporal-tracking training recipe. No model forward, SGD, GPU, or evaluation-as-training occurred.

| Result | Measured value |
|---|---:|
| Authentic source subset | 32 official COCO train2017 images;209 noncrowd source annotation instances |
| Canonical attempts / replacements | 1024 / **0** |
| Negative candidates drawn | 51/1024 =4.9805%; Wilson95%CI **[3.8082%,6.4892%]** |
| Accepted | 973/1024 =95.0195%; Wilson95%CI **[93.5108%,96.1918%]** |
| Rejected | 51, all `cross_image_physical_alias_unknown` |
| Accepted negatives | **0/973**; Wilson95% upper bound0.3933% (descriptive only) |
| Same-frame positives / temporal intervals | 973 / all0 frames; no temporal-motion evidence |
| Real active annotation/image representation | 197/209 instances;32/32 images across attempts |
| Source crowd quarantine | One annotation,900100000149 in image149; never eligible |
| Independent replay/storage audit | All1024 fresh candidate records/array hashes equal; all973 NPZ archives fully verified |
| Unit/contract tests | **25 passed**,7.89s; includes actual Windows2-worker spawn loader |

Intervals describe the frozen finite-cohort candidate process, not generalization, object-population coverage, or independent benchmark evidence. Repeated use of32 still images is not209 independent physical objects or973 videos.

### Immutable links and binding

- [Frozen manifest](<F:/Code/HiFT/artifacts/wave3-training/FROZEN-001/training_manifest.json>) SHA256 `5e1bd0151bcbdaa49fc40c4a7e21b11c1018c774cee793b7308d8e3d0749dabc`.
- [Frozen reservations](<F:/Code/HiFT/artifacts/wave3-training/FROZEN-001/reservations.json>) SHA256 `a4f0b91e0c94181f4f8dbc09a09762fa722d97cead728bd553c333447f508e7c`.
- [1024-candidate summary](<F:/Code/HiFT/artifacts/wave3-training/PREFLIGHT-001/summary.json>) and [every attempt/reason](<F:/Code/HiFT/artifacts/wave3-training/PREFLIGHT-001/candidate_ledger.jsonl>); ledger SHA256 `fd3b528dd4b85521e4e9968927d10ccf24b599d82e6c5a5be86552343db9b793`.
- [Independent all-array/replay verification](<F:/Code/HiFT/artifacts/wave3-training/PREFLIGHT-001/independent_verification.json>) and [full distributions](<F:/Code/HiFT/artifacts/wave3-training/DISTRIBUTIONS-001/statistics.json>).
- [API / usage / policy](<F:/Code/HiFT/docs/wave3_training_data.md>), [tests](<F:/Code/HiFT/tests/test_training_data_contract.py>).

The validator requires `status=frozen_preflight_only` and `production_training_authorized=False`. The lifecycle worker received the exact manifest/hash for **provisional** recipe binding; this result cannot open a real-SGD gate. B0/B1 inference, existing B2 graph, registry, previous wave records, and upstream were not edited.

## 1. Acquisition: inspected first, budgeted, TRAIN only

The [probe](<F:/Code/HiFT/artifacts/wave3-training/probe-001/probe.json>) used HTTPS HEAD and exact HTTP206 ZIP-directory ranges from [official S3 annotations](<https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip>) and [official TRAIN images archive](<https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip>). A complete instances_train2017 member is152,757,829 compressed bytes, larger than the initial128MiB training budget. That finding was reported **before** exceeding the limit. Lead explicitly authorized192MiB total, including the12,459,910-byte probe. No alternate dataset, val split, incomplete annotation fragment, certificate bypass, full images archive, or speculative download was substituted.

[Acquisition receipt](<F:/Code/HiFT/artifacts/wave3-training/acquisition-001/acquisition.json>) verifies:

- **171,734,757 payload bytes**,163.78MiB, including all prior probes/index reads; below201,326,592-byte approved cap. This counts HTTP body payload, not protocol/TLS overhead.
- Complete469,785,474-byte `instances_train2017.json`, with ZIP CRC validation by EOF and SHA256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`.
-32 authentic JPEG members, decoded against official width/height and CRC/hash. Original images and complete annotation member remain under [training source data](<F:/Code/HiFT/data/wave3-training/acquisition-001/>).
- [Image/annotation preselection](<F:/Code/HiFT/artifacts/wave3-training/acquisition-001/selection_before_images.json>) was written **before image acquisition**: smallest numeric train image IDs with positive finite noncrowd annotations, max32images. No image was selected by tracking outcome, inference, or target statistics.
- Each exact range, Last-Modified, member identity, cache hash, source URL, JPEG SHA, and license ID is retained. The official image license list and original image/Flickr URLs are recorded, **not a blanket legal opinion or invented permission grant**.

Upstream [configuration](<F:/Code/HiFT/third_party/HiFT/pysot/core/config.py#L160-L183>) requires prepared crop511 and train JSON for VID/GOT/YOUTUBEBB as well as COCO. This bounded adapter registers only authenticated COCO train2017. VID/GOT/YTBB source assets, corresponding prepared crops/metadata, licenses/provenance and overlap screening are absent in this scope; therefore they are explicitly unregistered rather than silently replaced with eval benchmarks or guessed temporal identities. This does not assert their official registration/download services are inherently broken.

## 2. Actual source path and retained geometry

The actual pinned [COCO preprocessing](<F:/Code/HiFT/third_party/HiFT/training_dataset/coco/par_crop.py>) crops source annotation boxes using SiamFC context0.5 and original-image mean padding, then writes JPEG search crop511. [COCO metadata generation](<F:/Code/HiFT/third_party/HiFT/training_dataset/coco/gen_json.py>) represents each image as a still-image pseudo-video with frame000000. [Dataset `_get_bbox`](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/dataset.py#L208-L224>) normalizes source dimensions around the crop center; [actual augmentation](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/augmentation.py#L31-L139>) supplies shift/scale/color/blur. The [experiment policy](<F:/Code/HiFT/third_party/HiFT/experiments/config.yaml#L59-L82>) is preserved for this source path: template4/.05/0/0/1, search64/.18/.2/0/1, gray0, candidatePneg.05.

[Source operations](<F:/Code/HiFT/src/hift_research/data_contract/source_ops.py>) execute unchanged pinned function/class AST bodies. Removed `np.float` is supplied by a **local proxy only**; NumPy/upstream globals and files are unchanged. Default source JPEG write/read is reproduced with explicit quality95. Pixel bytes equal actual default `cv2.imwrite` in the parity test. The audit subclass overrides only `_crop_roi` to record its geometry before calling the original implementation; tests require bitwise pixels, bbox, and subsequent NumPy RNG parity.

**Important retained mismatch:** actual511 pixel affine scale is **510/source_crop_side**, not511/side. Original centered labels use ideal127/context_side. Their scale is511/510 times the actual crop pixel scale (~0.1961% larger). The contract records both and does not silently repair this source geometry. Crop corners map to0 and510 in the affine test. Resizing does not create source-image detail.

[Pipeline](<F:/Code/HiFT/src/hift_research/data_contract/pipeline.py#L87-L193>) carries dataset, canonical identities, both source boxes, seed/pair type/replacement status on **every attempt including rejections**. Accepted attempts also carry both centered/augmented bboxes, preprocessing and augmentation affine matrices, JPEG/array hashes, visible source support, actual510-based scale, all shapes/dtypes/domains, and support/mask counts. Rejected-before-augmentation records have explicit null augmentation fields; no fabricated geometry is used. Positive empty class/regression support, invalid mask domains/shapes, nonfinite arrays, or wrong model-input types cause explicit rejection.

The frozen sampler is **uniform active image→uniform source instance per indexed seed**, not upstream shuffled/repeated four-dataset global picks. That bounded-subset deviation and lack of VID/GOT/YTBB mixing are frozen and disclosed; no full training-recipe equivalence is claimed.

## 3. Identity, leakage and rejected negatives

Canonical identity uses official **imageID+annotationID**, scoped to a visible source annotation instance. It does not label a new global physical object per photograph. The [negative validator](<F:/Code/HiFT/src/hift_research/corrections/positive_targets.py#L16-L44>) rejects missing/colliding source keys; the real-data path additionally quarantines different-image negatives whose physical alias relation is unknown. Same-frame positives are source-legitimate, not disguised temporal pairs. Same-image distinct noncrowd annotations are eligible only below source-bboxIoU0.9; source-ambiguous overlaps are rejected.

Every candidate retains the original0.05 prior draw. All51 observed negative draws selected different images and were rejected. **No resampling, relabeling, replacement, or forced negative prevalence was applied.** Accepted negatives are0, so using accepted examples as though they preserve5% negative training would be false. No real training was run.

Known eval separation was frozen **before1024 candidate execution**:

- All15 reserved source-folder families and known variants: bike1,bird1,boat7,building1,building5,car16,person22,truck1,uav1,uav2,uav4,uav6,uav7,wakeboard7,wakeboard9.
-2937 eval frames from the actual [Wave2 manifest](<F:/Code/HiFT/artifacts/acquisition/WAVE2-UAV10-PILOT-002/dataset_manifest.json>) and [new Wave3 manifest](<F:/Code/HiFT/artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json>), bound by SHA and per-frame SHA/pHash64.
- Training source dataset names must be COCO train2017; UAV123/UAV20L/UAV123@10fps/DTB70 cannot be registered as train sources. Only eval image content/lineage is used for separation; no eval GT boxes/flags become training labels.
-0 exact or pHashHamming≤4 eval matches among32sourceimages. The independent **source-to-source** audit checked all496pairs: minimum pHash distance20,0near pairs/0exact duplicate quarantines. This latter audit was post-preflight but before any training; it did not change the frozen cohort or counters.

SHA and pHash do **not** certify absence of semantic, transformed, cropped, transcoded, or unseen aliases across datasets. Those unknowns remain explicit, not mislabeled as a proven global-object-disjoint split. There are0real tracking videos. The frozen manifest is train-source-only and image-lineage-disjoint under known content evidence, not a proof of full video-training coverage or global physical disjointness.

### Separate real negative edge probe (not an extra favorable cohort sample)

Because no eligible negative happened to occur in the1024 draws, an explicit [EDGE-NEGATIVE-001](<F:/Code/HiFT/artifacts/wave3-training/EDGE-NEGATIVE-001/result.json>) test was run **outside the cohort**. Its selection rule was frozen before targets: smallest source image then annotation IDs with distinct noncrowd spatially disjoint boxes. Authentic image9 annotations1038967(category51) and1913551(category55) have source-boxIoU0. They traverse the same pinned crop/augmentation and robust target API. cls1,cls2,and regression weight are **all dense zero**, with finite correctly shaped real-image model inputs. Arrays and selection-before-target records are retained. This validates the source-legitimate negative branch, not its population prevalence or physical identity across images. It does not enter any1024 counts or repair the accepted-negative rate.

## 4. Required descriptive distributions

The [full audit artifact](<F:/Code/HiFT/artifacts/wave3-training/DISTRIBUTIONS-001/statistics.json>) contains mean and p01/p05/p25/p50/p75/p95/p99 quantiles, candidate/instance representation, all-cell weights, class-label counts, and zero support checks. Tables below are **candidate-weighted across973 accepted positives**, not unweighted209-object population estimates.

| Quantity | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Source bbox width(px) | 4.45 | 128.04 | 535.98 | 636.57 |
| Source bbox height(px) | 4.14 | 152.31 | 499.05 | 635.40 |
| Source bbox area(px²) | 35.60 | 16044.77 | 185726.86 | 326770.93 |
| Source aspect(w/h) | 0.141 | 0.908 | 2.392 | 9.152 |
| Augmented label width(px) | 16.72 | 61.74 | 97.09 | 144.75 |
| Augmented label height(px) | 12.58 | 66.26 | 104.25 | 131.23 |
| Augmented label area(px²) | 1526.35 | 3883.81 | 5083.91 | 5936.71 |
| Augmented label aspect(w/h) | 0.148 | 0.925 | 2.550 | 11.217 |
| Regression support cells/pair | 12 | 22 | 34 | 52 |
| cls1 active cells/pair | 14 | 31 | 52 | 56 |
| cls1 background cells/pair | 10 | 22 | 37 | 40 |
| cls1 ignored cells/pair | 65 | 90 | 100 | 107 |
| cls1 positive cells/pair | 4 | 9 | 15 | 16 |
| cls2 positive cells/pair | 14 | 34 | 56 | 80 |

Source/augmented ranges are marginal distributions; they do not imply that an enlarged bbox contains newly acquired source detail. Zero-offset temporal interval/motion for all973positives is a still-image limitation, not measured tracking robustness.

**All active regression-cell weights, not just per-pair maxima:**24090cells; min4.4643e-7, median0.50000039, P95/P99=1.5, max64. Source boundary weight64 appears in **153cells** (0.6351% of active cells) across55/973pairs. It is retained, not retuned; its effects on real optimization remain unmeasured. There are93643zero regression-mask cells. The973maps contain117733cells:8713cls1positive,21651background,87369ignored;36587cls2positive and117733dense cls2loss-mask cells. **Empty positive support count=0** separately for cls1,cls2,regression; collapsed fallback observed0times. Absence here does not erase the synthetic tiny-box crash/fallback evidence.

All masks have shape1×11×11; localization target4×11×11. Source-style audit arrays are float32 images/localization offsets and float64 class/weight/bbox arrays. The separate CPU collator explicitly converts accepted batch fields toFP32, retaining original audit dtype/hash records.

## 5. Production-facing bridge and verification boundary

[CandidateDataset](<F:/Code/HiFT/src/hift_research/data_contract/torch_adapter.py#L17-L38>) exposes len1024/indexed attempts and stores only manifest path/length across serialization. Pinned dynamic AST classes/image caches are constructed lazily per process, not pickled into Windows workers. [Reject-aware collate](<F:/Code/HiFT/src/hift_research/data_contract/torch_adapter.py#L41-L63>) emits CPU model inputs plus accepted/rejected identity records; all-rejected batches return `model_inputs=None`, never replacement data. This is a real accepted-input bridge, not authorization to ignore rejections.

Tests cover actual Windows `spawn` with2workers, pickle after pipeline initialization, replay identity, both accepted and rejected collations, local NumPy-alias isolation, JPEG/pixel/box/RNG parity, source geometry, fail-closed train/eval split/annotation identity/domain/hashes, immutable outputs, and acquisition budget/image caps. Scoped NumPy seeding is **serial per process**, not thread-safe. Six-worker asynchronous loader/checkpoint resume is untested; this report does not inherit the synthetic lifecycle worker's resume results as proof of this loader.

The [independent verifier](<F:/Code/HiFT/src/hift_research/data_contract/verify.py>) freshly recomputed **all1024 candidates**, reread/digested all973saved NPZ files with `allow_pickle=False`, checked exact arrays/shapes/finite values, authenticated annotation/member/selection/code provenance, budget ledger, eval-manifest hashes, and a mixed real accepted/rejected CPU-FP32 collation. It records Python3.13.0, NumPy2.4.4, OpenCV4.13.0, torch2.10.0+cu128, OpenCVthreads1, sourcegitHEAD `fba88e5b3452bf6e92980f14a52604d5d93eb0b7`, tracked dirty-diff hash, and upstream clean SHA `992aa7c1605360d607af84cb804b93cd8ee81caf`. Critical data-input files are individually hash-bound; later standalone verification/distribution/edge-probe modules do not change the frozen pipeline.

**Unresolved before any real training:** faithful four-dataset/video mixture; legal/source provenance review appropriate to intended use; canonical cross-image/cross-dataset physical aliases; a logged negative policy that does not silently zero all real negative prevalence; full production sampler/worker-resume validation; and controlled optimization/recipe acceptance. No claim of real-SGD readiness, benchmark gain, reproduction, novelty, or production-negative distribution is made. Unknown root temporary files reported by the lead were not created or deleted by this worker; all acquisition writes stayed in assigned training paths.
