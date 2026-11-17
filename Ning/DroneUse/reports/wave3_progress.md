# HiFT Wave 3 — bounded delivery and scientific decision

## Executive summary

- **No real-data optimization occurred.** No released checkpoint was fine-tuned; no substantive B2/R1 training, R1 architecture, baseline retuning, or upstream edit occurred.
- The authentic TRAIN preflight executed **1024 attempts:973 accepted same-frame positives,51 quarantined requested negatives,0 replacements**. The rejection contract works; **zero accepted negatives is not a usable training recipe**.
- The actual corrected B2 graph passed restricted synthetic CPU checkpoint/resume verification, including nine bit-exact comparisons. This is not production loader/CUDA resume or full-batch-equivalence evidence.
- **12 new complete sequences /2672frames per baseline:** B0 AUC **0.519714**, B1 **0.475797**. The adverse result is retained. Independent paired-sequence verification passed; B1 promotion is not supported.
- Crop availability limits selected episodes, but drift usually precedes crop loss. Contained failures persist. Source-information causality is not established. **Primary next step: continue data/recipe correctness work**, not a B2 pilot or R1 implementation.

[Committed compact evidence](<../experiments/wave3_validation.json>) contains measured distributions, all per-sequence ordinary scores, both physical-origin sensitivities, hypothesis assessments, gates and hash-bound references. Detailed specialist reports: [data contract](<wave3_data_contract.md>), [lifecycle](<wave3_lifecycle.md>), [expanded evaluation](<wave3_evaluation.md>).

## A. Starting state, constraints and provenance

Starting HEAD was **`fba88e5b3452bf6e92980f14a52604d5d93eb0b7`**, with a clean working tree. The earlier Wave2 implementation commit was not substituted for its actual delivery HEAD. Initial checks passed254tests in13.64s, dependency consistency, and immutable handoff verification. [Handoff receipt](<../artifacts/wave3-diagnostics/WAVE3-HANDOFF-001.json>) rehashes all five recorded Wave2 references without rerunning their experiments.

Immutable upstream remains **`992aa7c1605360d607af84cb804b93cd8ee81caf`**. The official tracker checkpoint remains SHA256 **`8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`**. B0/B1 configs and critical inference mathematics were not edited. Wave2 reports, predictions, profiles and reference artifacts remain unchanged. A small backward-compatible acquisition change adds optional selection metadata and correct cohort-size wording, not tracking behavior.

Measured ordinary manifests record the starting revision, dirty-diff/source-tree/per-file hashes, checkpoint/config/data identities, seed, environment and timings. Work ran in the existing Windows/Python3.13/torch2.10.0+cu128 environment; GPU workloads were serialized on the RTX5070Ti. This is not proof of equivalence to the historical2021 runtime. Later documentation is not retroactively attributed to an earlier measured source snapshot.

### Bounded acquisition

| New HTTP response payload | Actual | Approved bound |
|---|---:|---:|
| Expanded evaluation |221625794bytes (~211.36MiB)|256MiB|
| Training metadata/images, including probes |171734757bytes (~163.78MiB)|192MiB;32images|
| Combined |393360551bytes (~375.14MiB)|Both separate bounds satisfied|

The initial128MiB training limit was revised explicitly **before exceeding it**, after discovering the complete official training-instance metadata member alone required152757829compressed bytes. No full images archive/dataset or unrestricted fallback was fetched. Per-request URL/range/archive identity/member CRC/hash ledgers are preserved; full archive SHA is unknown where only ranges were downloaded. License/source metadata is retained, not represented as a blanket legal certification.

## B. W3-G1: canonical real-input contract

**PASS within the bounded authenticated COCO contract, not completion of the released four-dataset M1 recipe.**

The [frozen training manifest](<../artifacts/wave3-training/FROZEN-001/training_manifest.json>) SHA256 is `5e1bd0151bcbdaa49fc40c4a7e21b11c1018c774cee793b7308d8e3d0749dabc`. It binds32official train2017images,209eligible annotation instances,18critical source hashes, sampling policy and frozen evaluation reservations before the1024 attempts. One crowd annotation is quarantined. Authoritative image/annotation IDs establish an **instance within one source image**, not a different global physical object for every photograph.

| Contract result | Measured |
|---|---:|
| Attempted / accepted / rejected |1024 /973 /51|
| Requested negatives |51/1024=4.9805%; Wilson95%[3.8082%,6.4892%]|
| Acceptance fraction |95.0195%; Wilson95%[93.5108%,96.1918%]|
| Accepted negatives |**0/973**|
| Replacement / silent resampling |**0 /0**|
| Authentic image / annotation representation |32/32;197/209 across attempts|
| Temporal positives |All same-frame; interval and motion0|
| Empty positive cls1/cls2/regression support |0 /0 /0|
| Source weight64 |153/24090active regression cells;55/973positive pairs|

All51 requested negative pairs crossed images with unknown physical alias relations and were explicitly rejected, not relabeled or replaced. Candidate Pneg0.05 is preserved; the accepted cohort does **not** retain5% negatives. A separately frozen, explicitly controlled real negative branch probe uses two spatially disjoint official annotation instances in one image; its dense background labels/zero regression weight pass. It is **outside** the1024 cohort and does not repair its negative prevalence.

The production-facing path executes the pinned COCO source crop511/JPEG95, original centered-box and augmentation bodies, then corrected B2 targets. Both source and augmented boxes, affine geometry, seeds, masks, shapes/dtypes, support counts and array hashes are recorded. Actual preprocessing pixel scale is **510/source_side**; the source's ideal label scale retains the **511/510** mismatch rather than silently correcting it. Pixel/box/RNG parity is tested. Uniform indexed image→instance sampling is an explicit bounded-subset deviation from the original shuffled four-dataset sampler.

A lazy per-process [CPU Dataset/collator](<../src/hift_research/data_contract/torch_adapter.py>) supports actual Windows2-worker spawn, separates rejections and returns no fabricated minibatch when all candidates reject. Scoped NumPy seeding is serial within each process, not thread-safe. All1024 fresh candidate records and all973 stored NPZ arrays were independently replayed/verified; this is not asynchronous production-loader resume proof.

Known source-folder families/variants and all2937local evaluation-frame SHA/pHash values were frozen as reservations. No32training image matches an evaluation SHA or pHash distance≤4. All496source-source pairs were separately checked: minimum pHash distance20, no exact/near duplicate. These checks do **not** disprove semantic/transcoded/cropped/unseen cross-dataset aliases or official pretraining leakage. VID/GOT/YOUTUBEBB are explicitly unregistered here, not replaced by evaluation imagery. Full source/augmented size/aspect/area/support/weight distributions and caveats are in the [data report](<wave3_data_contract.md>).

## C. W3-G2: source-derived lifecycle

**PARTIAL overall; bounded synthetic lifecycle sub-gate PASS.**

[Source audit and actual-graph verification](<wave3_lifecycle.md>) derive SGD, LR schedule and BN policy from the pinned trainer/defaults/released YAML, not generic optimizer defaults:

| Item | Verified / proposed policy |
|---|---|
| Source global batch / epochs |220 across2GPUs;70epochs;600000candidate pairs/epoch|
| SGD |BASE_LR0.005; backbone0.1×; momentum0.9; WD0.0001; clip norm10|
| Effective grader LR at epoch0 /5 /69 |**0.0025 /0.005 /0.00005**|
| Epoch clock |Data epoch, not optimizer update; warmup normalizes by source normal.start_lr0.01|
| Freeze/BN before10 |Backbone frozen and BN eval; grader train|
| Unfreeze10+ |Layers3–5 weights/BN active; layers1–2 always frozen/eval|
| Explicit proposed deviation |Stable optimizer groups preserve grader momentum; source rebuild loses it|
| Proposed single-process batch |Physical4×55 accumulation; not equivalent to physical220/DataParallel|

Accumulation uses whole-window class/mask/weight denominators, actual final partial-window counts, and one clipping/SGD step per effective window. It does not average independently normalized microbatch losses. BN/dropout, objective normalization and device partitioning prevent claims of full-batch equivalence.

The [primary immutable CPU-b verification](<../artifacts/wave3-lifecycle/cpu-20261023-b/verification.json>) reports all nine comparisons true: sample/loss trace, model and BN buffers, momentum, scheduler/LR, sampler, pending-state completion, gradients, Python/NumPy/Torch RNG, and final partial step. Tests cover24interruption cases across epochs0/4/5/9/10/69; actual127/287 B2 graph replay is included, not replaced by toy-only evidence. Loading is restricted `weights_only=True`, validates source/config/data/optimizer/shape/dtype identities and rejects custom pickle objects. CUDA-used checkpoints and production asynchronous loader replay are not claimed supported. The12.66s CPU harness runtime is not training throughput.

A separate disposable backbone audit permits **only five missing scalar int64 BN counters initialized to0**, strict-loading all other tensors unchanged. All30original tensor hashes and the upstream binary SHA are preserved; no pretrained tensor entered synthetic optimization. Missing learned/running state or unsupported cumulative-average BN fails closed.

The [bound proposed config](<../configs/B2-wave3-proposed.json>) SHA256 is `e5a44406d3b05d67b87e9377bd9ba799c9844b2d606a862cd2509b3f3e593cb4`; it pins the authentic manifest/policy/reservations and18source hashes. **validated_real_data=false;launch_allowed=false;real_data_sgd_authorized=false.** Zero accepted negatives, no temporal/four-dataset representation, unaccepted production loader/BN/accumulation behavior and absent training authority keep G2/G5 partial.

Existing Wave2 resident synthetic profiles were reused, not rerun. Their31.19666ms physical4 full-step mean yields only a graph-arithmetic proxy:100effective220 updates≈171.58s, plus a248MiB crop-only host buffer. This is **not** a real pipeline wall-time bound, a pilot recommendation, or a launch grant; loading/augmentation/H2D/validation/checkpoints and changed accumulation semantics are excluded.

## D. W3-G3: expanded ordinary evaluation

**PASS for preregistration/execution/independent matched verification; not B1 promotion.**

The [preregistration](<../artifacts/wave3-evaluation/PREREG-001/plan.json>) froze at `2026-10-23T13:04:07.462267+00:00` before new images/inference. Exact additional complete segments: **bike1,boat7,car16_1,person22,truck1,building1,bird1_1,wakeboard9,uav1_1,uav2,uav4,uav7**. They use12distinct reserved source folders, excluding the three Wave2 folders. Bike1 has1029frames and uav1_1 has519; source/condition diversity was not reduced to only short clips. Coverage includes official LR8/FM4/SV11/FOC4/ARC5/SOB2 flags. Known siblings/30fps/UAV20L variants are reserved, not independent sequences.

All2672original JPEGs were CRC/hash/decode/order checked. Frozen B0/B1 used identical official weights/config identities/seed/native inputs. Initialization is raw first xywh plus `[0.5,0.5,0,0]`, **no subtract-one**; primary evaluation retains native GT numbers, initialization and unavailable rows. Success uses strict IoU>`0:0.05:1`, arithmetic mean of21ordinates; P20 uses≤20pixels; aggregation is equal-sequence. Annotation origin remains unresolved.

| New12-sequence primary metric | B0 | B1-score | B1−B0 | Paired sequence95% interval |
|---|---:|---:|---:|---|
| AUC |0.519713850|0.475796748|−0.043917101|[−0.084969793,−0.009358869]|
| P20 |0.651454563|0.588734593|−0.062719970|[−0.127665680,−0.008670634]|

The [independent scalar CPU audit](<../artifacts/wave3-evaluation/CPU-AUDIT-001.json>) verifies all24runs, pair/frame/initialization/source/config/seed identities, scalar metrics, failure counts and driver macro. Intervals use10000paired whole-sequence resamples, seed20261023; they describe this selected panel, not random-population or retraining uncertainty. SOB and long-sequence strata each have2clusters, so intervals are null/descriptive. Overlapping attributes are not independent causal strata.

**B1 underperforms on this panel.** Valid-GT zero-overlap tracked frames are434→539. With all33unavailable-GT protocol failures retained, total failures are467→572. No row or unfavorable sequence was dropped/replaced, and no retuning followed. Mathematical score-domain correction does not establish empirical accuracy gain or probability calibration. The [full per-sequence AUC/P20/failure table](<wave3_evaluation.md#all-preregistered-sequences-retained>) retains favorable and unfavorable deltas, including uav1_1 AUC improvement with more zero-overlap frames.

The original3-sequence Wave2 macro remains0.552221506/0.551335666AUC. Descriptively combining those unchanged outputs with the new12 gives15-sequence AUC **0.526215381/0.490904532**, P20 **0.659902389/0.609726413**. This is not a new run, full benchmark, or longitudinal comparison of different cohorts. Full M2 remains partial.

## E. W3-G4: failure attribution and separate oracle controls

**PARTIAL: diagnostics completed, but representation versus optical/source-information causality remains unresolved.**

The [frozen diagnostic plan](<../experiments/wave3_diagnostic_plan.json>) precedes new tracking outcomes. All15sequences/2937frames are retained; after15initializations and34unavailable-GT rows, physical comparisons use2888valid tracked frames. Current GT is joined **only after ordinary inference**. Physical-origin1 is a declared assumption, always repeated at0, never substituted for primary native metrics.

Definitions: full/partial/outside describe actual source-crop containment; **fully available** additionally requires joint source-image overlap. Zero IoU=0, low IoU<0.1, reliable≥0.5. Native minimum-side bins0/16/32, nominal resized bins0/32/64, motion bins0/0.5/1, elongated aspect≥3 are fixed. Bbox containment does not establish visibility through occlusion. NaN rows remain absent; no interpolation or fabricated transitions crosses a GT gap.

### Crop availability versus drift

| Physical origin1; valid frames | B0 frames /zero-overlap | B1 frames /zero-overlap |
|---|---:|---:|
| Full crop containment |2337 /48|2350 /116|
| Partial |65 /32|46 /23|
| Outside |486 /388|492 /433|
| Full-crop low-IoU cases |119|159|

There are no source-bbox-off-image cases under origin1; origin0 has4partially clipped boxes, explicitly recorded. Neither assumption identifies occlusion. At origin0 the B0 category counts are2336/71/481, zero counts48/36/383; B1 counts2351/45/492, zero counts116/23/433. The qualitative conclusion is unchanged, but exact physical counts are not disguised as primary native metrics.

Per-sequence mean share of zero-overlap failures not fully contained is **0.7877** for B0,2000sequence-cluster95%[0.4656,0.9677], and **0.8905** for B1,[0.7938,0.9707], over6failure-containing sequences each. These wide selected-panel intervals are not frame-IID confidence or proof of the initiating cause.

Of **35 B0 /51 B1 observed crop-loss transitions**, **33 /50** already had preceding IoU<0.1; none followed a≥0.5prediction. Six outside observations after unavailable-GT gaps per baseline are explicitly **not** counted as observed transitions. Thus prolonged crop absence often follows earlier drift. Availability matters, but labeling all later outside failures as evidence that crop policy initiated them would be circular.

### Contained response and policy limitations

Fully contained low-IoU failures remain119(B0)/159(B1); zero-overlap failures48/116. Across full/partial cases with non-absent input, the fixed candidate-coverage diagnostic labels171/172B0 and189/198B1 low-IoU cases as lacking a decoded candidate reaching IoU0.5. Remaining cases separate scoring/selection and update-policy limitations. This is **not** pure feature-fusion identification: regression/grid/scale, occlusion, history and policy still confound it. GT-site/background raw-score contrast, top-two margin, class-mass entropy proxy and candidate boxes are saved; they are not calibrated information measurements.

An additional **explicitly nonpreregistered secondary control**, the [minimum-size addendum](<../experiments/wave3_diagnostic_addendum.json>), was frozen after ordinary aggregate results but before retrospective analysis: the original10px output minimum mechanically bounds achievable IoU by `GT_area/(max(10,w)*max(10,h))`. Twelve valid frames per baseline cannot reach0.5under this bound, but only1B0/0B1fully available frames are affected under origin1. Such frames remain in all primary/failure counts and are excluded only from secondary contained quality comparisons. No baseline or sample was changed.

Across2922tracked steps, B0 selected score lies outside[0,1] in2569steps and interpolation LR in182; B1 has0/0. Holding each B0 response/state fixed, the two scoring formulas select different indices in1682steps (1827on B1 states). This demonstrates substantial score/prior-policy interaction, **not** a causal decomposition of accuracy loss. B0's larger score range and preexisting tuned window/update strengths are alternative explanations; B1's bounded range does not rescue its adverse measured accuracy.

### Source information beyond bbox size

The small-native/large-resized group has317B0frames;198are fully available and only5have zero overlap. B1 has268frames;193fully available,10zero overlap (origin0:192/8). Enlarged crop footprint is not newly acquired source detail. This expanded evidence does not reproduce a universal small-source failure pattern.

Within-sequence comparisons control coarse containment, footprint, motion and aspect and require≥5small/≥5larger observations:5eligible B0cells and7B1cells remain. Time, appearance and state history remain confounded; cells are not independent videos. Beyond-size proxies use original pixels: intensity variance, Sobel energy, Laplacian variance and target/surround64-bin BGR histogram distance. These are **unvalidated texture/blur/clutter proxies**, not visibility, semantic distractor or information-loss measurements.

There are **zero matched B0 quality cells and one B1 cell**, uav4 with6failed/5reliable frames. Its failed median Laplacian variance is1849 versus685for reliable frames, Sobel energy39285 versus21009, histogram distance0.422 versus0.323—opposite a simple lower-texture/lower-contrast failure story. One temporally correlated cell cannot establish or refute general source-information causality. H3 remains insufficient, not a selected explanation for every contained failure.

### Offline GT-assisted 2×2 crop probes — NOT tracking scores

The [oracle manifest](<../artifacts/wave3-diagnostics/ORACLE-002/manifest.json>) and [per-probe records](<../artifacts/wave3-diagnostics/ORACLE-002/probes.jsonl>) use96preselected evenly spread cases:59full,17partial,20outside. Selection uses containment, not favorable outcomes. Each case runs observed/GT center × observed/enlarged side at both origins: **768frozen B0 frame probes**. Source side≤4096;0budget exclusions. Template is initialized from the immutable first frame; ordinary prior state is held fixed and **no probe state propagates**. Same side isolates a translation intervention; side changes also change resized scale/context. These controls offer privileged conditional headroom, not a formal optimum or deployable AUC upper bound.

| Origin1 conditional emitted-probe mean IoU (NOT tracking metric) | Observed center/side | GT center/same side | Observed center/enlarged side | GT center/enlarged side |
|---|---:|---:|---:|---:|
| Full,59cases |0.6366|0.6757|0.6366|0.6757|
| Partial,17cases |0.0390|0.4032|0.0576|0.3534|
| Outside,20cases |0.0022|0.3695|0.0022|0.1874|

On outside cases, GT-recentering at the same side makes18/20fully available;12have some candidate≥0.5and7emit≥0.5. Merely widening around the observed center makes20/20fully available but0emit≥0.5. Widening/recentering also lowers median resized target footprint; even both privileged controls do not solve every case. That supports a crop-position/scale/context interaction and contradicts an unqualified "containment alone guarantees recovery" claim. It does not establish missing native optical information or a novel fusion remedy.

All192observed-condition raw-response reproductions are **exact (maximum difference0)** and emitted boxes pass literal smoothing/clipping replay. Model-state SHA before/after is identical, `28e7c2b4e3263292862013e40a6c7ac527d1c94b46038c3408e2b35936bfe982`. Both origin summaries, all failures and candidate/selection statistics are retained separately from ordinary runs. No GT-assisted result enters the ordinary controller, primary scores or training.

### Preserved corrections/failures, not silently overwritten

- The first oracle attempt stopped before any probe: recorded Wave2 observer-script hash differed from the final Wave2 delivery script. [Failed receipt](<../artifacts/wave3-diagnostics/ORACLE-001/failure.json>) is preserved. A [narrow hash-pinned adapter](<../src/hift_research/wave3_diagnostics/observer_compat.py>) reverses **exactly** the historical added input guard and grouped observer ratio, reconstructs the old script hash, and requires the immutable524-crop audit/run-manifest bindings. Unknown changes still fail. Inference/model source checks were not waived.
- The first CPU retrospective overcounted transitions after absent-GT gaps. The final [corrected CPU retrospective](<../artifacts/wave3-diagnostics/RETROSPECTIVE-002/summary.json>) explicitly excludes them and adds policy/source-image support audits; the previous result is retained, not overwritten. Its oracle-case SHA matches the already completed frozen probes **exactly**; no tracking or GPU oracle was rerun for this CPU correction.

## F. Gates and independent hypothesis assessments

| Gate | Status | Interpretation |
|---|---|---|
| W3-G1 real-data contract |**PASS, bounded**|Authenticated COCO path/identity/quarantine/arrays/replay; not full four-dataset M1|
| W3-G2 lifecycle |**PARTIAL**|Synthetic core passed; real distribution, CUDA/production-worker recipe not accepted|
| W3-G3 expanded evaluation |**PASS**|12preregistered matched complete sequences independently audited; B1 adverse|
| W3-G4 attribution |**PARTIAL**|Availability and contained failure evidence measured; H2/H3 causality not separated|
| W3-G5 pilot readiness |**PARTIAL / NOT READY**|Zero accepted negatives, still-only data, unaccepted recipe; launch disabled|

| Hypothesis | Assessment | Supporting and counter/confounded evidence |
|---|---|---|
| H1 crop containment |**Supported**|Unavailable target pixels limit selected episodes and same-side recentering helps; drift usually precedes loss, widening alone fails. Not proven initiating/universal cause.|
| H2 representation |**Plausible but unconfirmed**|Contained low-IoU/candidate-coverage failures persist; regression/grid, score/update priors, occlusion, scale and history remain alternatives. No isolated fusion defect proved.|
| H3 source information |**Insufficient evidence**|Native-pixel proxies computed, but comparable quality cells are0/1 and contained small targets often succeed; weak proxies and few independent sequences cannot isolate optical information.|

These hypotheses are not mutually exclusive, and no winner is manufactured. There is **no R1 go decision**, accuracy/novelty claim, causal proof of source-resolution fusion, full benchmark reproduction, or real-trained B2 model.

## G. Exactly one primary next step and verification

**Choice2: continue data/recipe correctness work.** Obtain/freeze authentic temporal training sources with author-provided object IDs and known-distinct negative tracks; explicitly declare alias-safe sampler deviations, rerun bounded class/distribution preflight with attempts/acceptance/rejections separate, and validate production loader/recipe behavior. Unknown cross-image/cross-dataset identity must remain quarantined, not invented. This is the prerequisite work, **not a request to start a pilot**. Real-data SGD, substantive B2 training and R1 remain prohibited until separately authorized and the data/recipe gate is actually accepted.

Implementation and compact evidence are committed as **`506a8a8`**; the final documentation/registry commit follows it. Integrated verification: **362tests passed in24.58s**, including existing254tests and new contract/lifecycle/evaluation/diagnostic tests. A final full-suite rerun after evidence/registry integration also passed **362tests in38.50s**; no test was dropped. Dependency check reports no broken requirements. No new packages were installed for these results; OpenCV already in the pinned base environment is now also declared for data/test extras. [Final preservation verification](<../artifacts/wave3-diagnostics/FINAL-VERIFICATION-001.json>) **passed21Wave3 receipt references,5Wave2 references and14baseline files byte-exact to the starting commit**, plus checkpoint, clean pinned upstream and canonical code/config bindings. New report/evidence/registry records are committed without committing images, NPZ arrays or model weights. Exact ending HEAD/cleanliness are supplied in the delivery message to avoid a self-referential commit hash in this report.

The linked raw artifacts and source data remain ignored local evidence under the existing workspace; compact measured evidence is committed. Preserve those directories in any transfer: a clean Git tree alone is not a data/checkpoint backup. All required workers completed and their background jobs were collected; no GPU/training job remains running.
