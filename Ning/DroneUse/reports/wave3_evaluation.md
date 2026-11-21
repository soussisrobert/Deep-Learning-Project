# Wave3: preregistered expanded ordinary B0/B1 evaluation

**Measured outcome:** all24ordinary runs completed; independent G3 matched-identity/metric audit **passed**. On the fixed12-sequence panel, B1-score was **worse overall**, not promoted: macroAUC `0.519714→0.475797`, precision@20px `0.651455→0.588735`. Both paired sequence-bootstrap intervals for B1−B0 lie below zero within this selected panel. This is **not** an accuracy/stability acceptance, training approval, full-benchmark, or population claim.

## Identity and scope

This worker froze and acquired **12 additional complete official UAV123@10fps segments**, distinct from the three Wave2 source folders. No Wave2 inference was rerun. No training, checkpoint replacement, score retuning, oracle controller, or source-resolution intervention is part of this ordinary evaluation.

The immutable [preregistration](../artifacts/wave3-evaluation/PREREG-001/plan.json) was created at `2026-10-23T13:04:07.462267+00:00` **before new image acquisition or inference**; SHA-256 `157f3ce248d4d65ceb37d89d25c5dc96a205153fefa66d674c79b001a7781a47`. It records the exact source members/ranges, all official attribute flags, original config/attribute CRC/SHA evidence, baseline/config/checkpoint/critical-source identities, seed, invalid-frame and coordinate conventions, output roots, and statistical analysis.

### Fixed panel and source-folder reservations

| Official segment | Entire reserved source-folder lineage | Frames | Role |
|---|---|---:|---|
| bike1 | bike1 | 1029 | Longer trajectory / drift exposure |
| boat7 | boat7 | 179 | Boat |
| car16_1 | car16 | 139 | Car; all sibling segments reserved |
| person22 | person22 | 67 | Person |
| truck1 | truck1 | 155 | Truck |
| building1 | building1 | 157 | Building |
| bird1_1 | bird1 | 85 | Bird; all sibling segments reserved |
| wakeboard9 | wakeboard9 | 119 | Human water sport |
| uav1_1 | uav1 | 519 | Second longer trajectory; all siblings reserved |
| uav2 | uav2 | 45 | UAV |
| uav4 | uav4 | 53 | UAV |
| uav7 | uav7 | 125 | UAV |

Total **2672 frames**, not a full123-sequence benchmark or random sample. Source-folder diversity and deliberate inclusion of longer segments were specified before tracking. Size still affects selection; do not generalize a selected panel as unbiased population evidence. Reserve sibling segments and 30fps/UAV20L variants from these same source folders against training. Known source folders are minimum established lineages, not proof that all cross-dataset/transcoded aliases are absent.

Official12-bit order `SV ARC LR FM FOC POC OV BC IV VC CM SOB` comes from the preserved author [annotation definition PDF](../artifacts/acquisition/annotation_details.bin), SHA-256 `a78fd08851e395cbe9c9836a68b498de8b82fac2be00365a89a6688d1400cb5e`, not from target boxes or observed failures. Official sequence-level coverage: LR8, FM4, SV11, FOC4, ARC5, SOB2; longer segments with>=500frames2. Attributes are inherited official labels, not per-frame event annotations or inference features. SOB and long-trajectory strata contain only2 independent sequences; preregistered uncertainty intervals for these strata are null/descriptive rather than misleading narrow intervals. No sequence was replaced based on performance.

## Authentic data, bounded transfer and GT boundary

The publisher-linked official10fps Drive archive and verified Range/ZIP64 cache were reused, not a mirror or newly resampled dataset. Original archive identity remains4,688,528,930bytes, Last-Modified `Fri,19Dec2025 23:52:28GMT`; full archive SHA is unknown because it was not downloaded whole.

- [Metadata/attribute preregistration ledger](../artifacts/wave3-evaluation/PREREG-001/metadata_network_ledger.json):173,722new response payload bytes.
- [Acquisition](../artifacts/wave3-evaluation/ACQUISITION-001/acquisition.json) and [ledger](../artifacts/wave3-evaluation/ACQUISITION-001/network_ledger.json):221,452,072new response payload bytes, including selected images and tiny original metadata. Preregistered contiguous image-span estimate was221,436,184bytes.
- Combined new payload **221,625,794bytes (~211.36MiB)**, below the initially approved256MiB image budget. No budget extension or unbounded full archive fallback was needed.
- Every one of2672original JPEGs was fully decoded, exact official six-digit IDs/order/range/GT counts checked, every frame hashed, ZIP member CRC checked, and original GT bytes retained. Full-NaN absent rows remain in position; malformed partial-NaN/nonpositive-size rows fail rather than being repaired.
- [Dataset manifest](../artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json) SHA-256 `f87d13a05ed34916530b594774eff60498bddbc65cce14beed358c9d8a3be547`; root [prepared data](../data/wave3-evaluation/SOURCE-001/prepared/). It contains offline evaluation GT and must never enter an inference controller.
- Separate [ordinary GT-free input receipt](../artifacts/wave3-evaluation/INPUTS-001/input_receipt.json) SHA-256 `902a0f621eb9145f34ede99c1028315fa4c901914d994de97d154f98fdb963d3`; per-sequence inputs contain only ordered image paths and initialization, never later GT or attribute-driven decisions.

Both old Wave2 and new Wave3 source identities/frame hashes/paths were sent to the training source-audit worker before training freeze, for its independent SHA/pHash reservation screen. This is data separation, not a claim that perceptual hashing proves semantic or pretraining leakage freedom.

## Frozen protocol and execution

Both ordinary baselines use the same official full checkpoint `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`, upstream `992aa7c1605360d607af84cb804b93cd8ee81caf`, seed0, exact fixed config hashes and critical inference source hashes in the preregistration. B1-score is the existing inference-only sigmoid interpretation correction; no other changes are bundled into it. The normal shared runner and inference modules were not edited by this worker.

Primary evaluation preserves raw annotation numeric coordinates and literal upstream first rectangle `rawGT+[0.5,0.5,0,0]`, **no origin shift**, initialization included. All source absent rows count as failures in retained denominators for both baselines. Success is strictIoU>threshold at the upstream21-point grid and its arithmetic ordinate mean, precision is<=20px, sequence weighting is equal macro. Normalized precision remains supplemental. Source author coordinate origin remains unresolved; retrospective physical diagnostics explicitly assume origin1 for MATLAB consistency, separate from native-numeric primary scores.

The lead granted exclusive GPU token `eval-wave3-exclusive-001` only after CPU tests and source/data validation. Ordinary runs are serialized by the unchanged shared matched driver with alternating baseline-first order. [Matched run root](../artifacts/wave3-evaluation/MATCHED-001/) stores each sequence's B0/B1-score manifests, predictions, diagnostics and raw policy grids. GT is joined only offline. The oracle/teacher-forcing diagnostic belongs to the lead under a **separate output identity**, not this ordinary cohort. Desktop GPU latency is descriptive telemetry, not a controlled speed-advantage comparison.

## Preregistered independent CPU audit

[Independent audit implementation](../src/hift_research/wave3_evaluation/audit.py) separately recomputes scalar rectangle intersection/union and center precision directly from saved CSVs, checks them against saved metrics, and additionally invokes the common paired identity checker. It verifies baseline/config/checkpoint/source/seed/frame content/order and submitted initialization, paired complete frames, prediction hashes, diagnostics and policy-grid coverage. All adverse B1 deltas and both baselines' failures remain in the cohort.

Per-sequence output includes success/precision, zero-overlap episodes/counts excluding init, measured-valid-GT failures separately from absent-GT protocol failures, ordinary source-crop geometry/padding/resize summaries, retrospective physical containment under the declared origin assumption, and evidence hashes. Resizing is not treated as additional source detail.

Primary uncertainty is a **paired whole-sequence-cluster percentile bootstrap**,10000replicates, seed20261023,95%intervals. No frames are independently resampled and the two baselines are never independently resampled. All12sequences receive equal macro weight; the1029-frame sequence does not dominate the primary estimate. Overlapping official attribute strata LR/FM/SV/FOC/ARC/SOB and>=500frame strata are descriptive, not causal/multiplicity-adjusted hypotheses. Strata with fewer than3clusters have no interval. Intervals describe sensitivity within this deliberately chosen panel, not random-population, checkpoint-training, or retraining uncertainty.

## Reproducible commands

```bash
.venv/Scripts/python.exe scripts/wave3_evaluate.py freeze --output artifacts/wave3-evaluation/PREREG-001 --inventory artifacts/acquisition/uav10_content/inventory.json
.venv/Scripts/python.exe scripts/wave3_evaluate.py acquire --plan artifacts/wave3-evaluation/PREREG-001/plan.json --inventory artifacts/acquisition/uav10_content/inventory.json --data-root data/wave3-evaluation/SOURCE-001 --output artifacts/wave3-evaluation/ACQUISITION-001
.venv/Scripts/python.exe scripts/wave3_evaluate.py inputs --plan artifacts/wave3-evaluation/PREREG-001/plan.json --dataset-manifest artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json --output artifacts/wave3-evaluation/INPUTS-001
# GPU only with explicit lead allocation, after CPU/source/data checks:
.venv/Scripts/python.exe scripts/wave3_evaluate.py track --plan artifacts/wave3-evaluation/PREREG-001/plan.json --dataset-manifest artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json --inference-dir artifacts/wave3-evaluation/INPUTS-001 --output artifacts/wave3-evaluation/MATCHED-001 --gpu-token eval-wave3-exclusive-001
.venv/Scripts/python.exe scripts/wave3_evaluate.py audit --plan artifacts/wave3-evaluation/PREREG-001/plan.json --dataset-manifest artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json --runs artifacts/wave3-evaluation/MATCHED-001 --output artifacts/wave3-evaluation/CPU-AUDIT-001.json
```

Outputs are immutable; examples showing existing paths are provenance of the performed run, not commands to overwrite it. Reproduction needs fresh output paths and a new lead GPU allocation. Failed attempts must remain available; do not overwrite them or replace source rows/sequences to improve outcomes.

## CPU fixture evidence

Before GPU activation:

```text
.venv/Scripts/python.exe -m pytest tests/test_wave3_evaluation.py tests/test_data_acquisition.py tests/test_evaluation.py
67passed in1.19s  (13new Wave3 +19acquisition +35evaluation)
```

Fixtures establish independent metric/absence/boundary agreement, paired cluster resampling and deterministic unfavorable deltas, insufficient-cluster intervals, actual source-crop padding arithmetic, known source-folder/Wave2 overlap rejection, GT-free native initialization, immutable outputs, config drift and even rehashed-GT-input leakage rejection, and backward-compatible old acquisition defaults. Generated fixtures are not benchmark scores.

## Measured ordinary outcomes

All24ordinary inference runs completed successfully,2672frames per baseline. Exclusive GPU token was released to the lead immediately after driver exit; this worker performed only CPU work afterwards. No ordinary run was restarted or replaced based on the results. The checkpoint loader emitted a nonfatal NumPy dtype-align deprecation warning; checkpoint hash/strict tensor-key loading still passed. The modern Torch runtime is not a demonstrated numerical recreation of the historical training/inference environment.

### Independent G3 evidence

[Full immutable CPU audit](../artifacts/wave3-evaluation/CPU-AUDIT-001.json), SHA-256 `99b552e312f4cfcf04e047520d4a5892cf515879d71bd0e98770b90e7787a6a8`; [compact evidence receipt](../artifacts/wave3-evaluation/SUMMARY-001.json). **G3 verification passed**: separate scalar metrics, exact paired identities and prediction hashes, frozen config/checkpoint/critical-source/seed, raw+.5 submitted initialization, complete diagnostic/grid counts, failure counts, and the driver's equal-sequence aggregate all agreed. Verification pass is not B1 promotion or training approval.

Inference manifests record project revision `fba88e5b3452bf6e92980f14a52604d5d93eb0b7`, explicit dirty-source evidence, original upstream SHA, runtime hardware/software, checkpoint/config identities, frame hashes and timings. Example [first ordinary run manifest](../artifacts/wave3-evaluation/MATCHED-001/bike1/B0/manifest.json) records dirty diff SHA `4b992507d5a67ddd5f67b372b5d7dfd3bb84dbf305b2f635b9551bd50a4c33be`; all paired critical inference source/config hashes were independently checked against preregistration. Worker changes to the old acquisition adapter were limited to optional backward-compatible selection metadata/accurate cohort-size wording; no baseline mathematics or runner was edited.

| Primary equal-sequence metric (0–1) | B0 | B1-score | B1−B0 | Paired12-sequence percentile95% interval |
|---|---:|---:|---:|---|
| SuccessAUC |0.519713850|0.475796748|−0.043917101|[−0.084969793,−0.009358869]|
| Precision@20px |0.651454563|0.588734593|−0.062719970|[−0.127665680,−0.008670634]|

These are preregistered paired whole-sequence bootstrap results:10000replicates, seed20261023. They describe sensitivity within this deliberately selected12-cluster panel, **not** random-population or retraining confidence. On this panel B1 loses4.39AUC percentage points and6.27precision percentage points; therefore a robust ordinary B1 accuracy/stability advantage is **not supported**. Do not replace this unfavorable result with the mathematical interpretation argument, retune B1 on these sequences, or attribute a separate oracle/teacher-forcing diagnostic to ordinary B1 performance.

### All preregistered sequences retained

Zero-overlap counts exclude initialization and include source absence as explicit protocol failures. Precision/AUC retain initialization and all expected rows.

| Sequence | Frames | B0AUC | B1AUC | B0P20 | B1P20 | B0zero | B1zero | AbsentGT, both baselines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|bike1|1029|0.726086|0.716785|0.919339|0.889213|0|0|0|
|boat7|179|0.808194|0.810854|1.000000|1.000000|0|0|0|
|car16_1|139|0.704351|0.585132|0.834532|0.791367|0|0|0|
|person22|67|0.834399|0.840085|1.000000|1.000000|0|0|0|
|truck1|155|0.716743|0.518894|0.567742|0.329032|0|0|0|
|building1|157|0.835305|0.832878|1.000000|1.000000|0|0|0|
|bird1_1|85|0.155182|0.034734|0.364706|0.058824|32|80|19|
|wakeboard9|119|0.772309|0.780312|1.000000|1.000000|0|0|0|
|uav1_1|519|0.098174|0.138912|0.117534|0.181118|286|327|4|
|uav2|45|0.178836|0.147090|0.311111|0.311111|32|31|5|
|uav4|53|0.178796|0.132075|0.358491|0.264151|34|39|0|
|uav7|125|0.228190|0.171810|0.344000|0.240000|83|95|5|

Both failures are preserved: among2660tracked frames excluding the12initializations, B0 has467zero-overlap protocol failures and B1 has572. Of these,33per baseline are unavailable-GT rows, **not measured tracking errors**; measured zero overlap on valid GT is434(B0) versus539(B1). Positive and negative sequence deltas coexist: uav1_1AUC improves even while its zero-overlap count increases. No sequence was dropped, and no single summary establishes a mechanism.

Per-sequence ordinary crop size/resize/padding telemetry and separate retrospective physical-containment summaries are in the compact/full receipts. Original raw policy grids remain under the ordinary run root for the lead's separately identified diagnostics; they were not consumed by a new ordinary controller. Official condition-stratum outputs are retained in the full and compact receipts. **SOB(n2)** and **long>=500frames(n2)** have `percentile_95_interval:null` and `descriptive_only_fewer_than_three_sequences`; no pseudo frame-IID interval was substituted.

Final worker CPU rerun after audit hardening and report completion: **67passed in1.34s**, and `git diff --check` clean. The audit process exited0 with `Independent paired audit PASS:12sequences/2672frames;deltaAUC=-0.043917101`. The committed report carries compact measured evidence even though original data/full raw run artifacts remain ignored local files; artifact availability must be preserved in any handoff package. The separate oracle/origin/source-resolution diagnostics and any training decision remain lead-owned and are not accepted by this audit.
