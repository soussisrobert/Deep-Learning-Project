# Wave 2 — bounded correctness, authentic pilot evaluation, and feasibility

Measured on 2026-10-23, continuing the existing checkout. **No project reinitialization, R1 architecture, full-scale training, or trained B2 accuracy result.** Published results, these local measurements, mathematical tests, and future experiments remain distinct.

## Gate decision

| Gate | Status | Exact scope |
|---|---|---|
| G1 — correctness | **PARTIAL** | Corrected target/loss/identity APIs and actual synthetic graph pass; production identity-aware loader, retained edge weighting, schedule/resume and historical-runtime fidelity are not closed. |
| G2 — authentic complete sequence | **PASS** | Three complete publisher-distributed UAV123@10fps sequences, 265 original decoded frames, exact config ranges/GT alignment, hashes and absent-row handling. Not all123 sequences. |
| G3 — matched comparison | **PASS** | Both released-checkpoint baselines completed all265 frames under matched inputs/config/seed/software, with per-frame comparison. Passing this gate does not mean B1 improved. |
| G4 — training feasibility | **PASS** | Actual corrected graph, bounded FP32 forward/backward/clipped-SGD at batches1/2/4, no OOM, finite active gradients. Not a production training recipe or real-data throughput test. |
| G5 — source-support diagnostic readiness | **PARTIAL** | Retrospective geometric support, footprint, motion, containment, localization and failure records exist; origin/attribute granularity and mechanistic/confounding evidence remain insufficient for R1. |

**Milestones:** accept the delivered Wave2 evidence package; do not accept full M1 or full M2 as complete. B2 real-data training and R1 development remain gated.

## A. Repository and verification

- Starting commit: `5ce4a432aaa3027c7a45d4c678ae04e27a20223a`.
- Ending implementation/evidence commit: `2369156e280920f81d31c10f0720af83428f2409`. The subsequent documentation/handoff commit is the delivery HEAD, whose exact SHA is supplied in the completion message; recover it with `git log -1 --format=%H -- reports/wave2_progress.md`.
- Upstream remains immutable at `992aa7c1605360d607af84cb804b93cd8ee81caf`; all six isolated patches jointly pass `git apply --check`. None is applied to released-checkpoint inference.
- Initial108 tests passed before work. Final integrated suite: **254 passed, no skipped tests**, measured13.60s before handoff documentation; the final release rerun also passed **254 tests in14.20s**, with `pip check`, all six patch checks and upstream cleanliness passing. Restricted checkpoint loading retains the disclosed NumPy legacy-metadata deprecation warning, not unsafe pickle fallback.
- Changes: corrected target/identity/masked-loss APIs; isolated actual B2 graph and profiler; historical-formula compatibility probes; bounded official ZIP acquisition/validation; native-numeric GT-free inputs; passive policy observation; matched real evaluation and retrospective diagnostics; tests, provenance, reports and registry.
- Pillow12.1.1 is pinned for full image decoding in the isolated environment and dependency files. `pip check` passed. No global Python downgrade or historical torch installation.
- [Committed compact evidence](../experiments/wave2_validation.json) references immutable acquisition, inference, compatibility and profiling manifests. Original images/checkpoints/full local run artifacts remain ignored, not silently added to git. Final release checks require a clean tracked working tree and clean upstream.

Measurements were made from the starting commit plus **recorded dirty source-content hashes**, not falsely attributed to a clean final commit. Profiling also saved exact source snapshots. Inference-critical hashes agree within each B0/B1 pair. Subsequent target changes only strengthen invalid-input guards, outside timed target construction; [training provenance notes](wave2_training.md) disclose the exact hash difference. The final observer also uses the literal original grouped crop-size ratio. A separate [geometry re-verification](../artifacts/audits/WAVE2-GEOMETRY-001.json) confirms **all524 saved tracking crops** already match the pinned tracker exactly. These checks do not overwrite prior runs.

## B. M1 correctness and compatibility

Detailed evidence: [M1 audit](wave2_m1.md), [compatibility audit](wave2_compatibility.md), [training composition tests](../tests/test_training_profile.py).

### Closed within the explicit corrected API

1. **Collapsed positive support:** unchanged source crashes reproduced for six cases, including tiny objects and a physically55×56-pixel edge-clipped box. Minimum existing-grid-cell supervision is an explicit B2 fallback; decoded nonsaturated boxes recover the input. Eighteen ordinary-positive controls preserve arrays **and subsequent RNG values bit-for-bit**.
2. **Contradictory negatives:** actual source can sample the same object as a negative. New canonical API rejects missing, malformed or equal `(dataset,video,track)` identities before labeling. Distinct known negatives retain dense background labels and zero localization weights. No hidden resampling or negative-frequency change.
3. **Empty/singleton/numerical masks:** connected empty sums avoid ignored NaN/Inf poisoning; active values are validated. Regression is masked **before inverse tanh**, and ignored classification logits **before log-softmax**, so backward does not encounter `0*NaN`. Ordinary finite loss values/gradients are checked against source.
4. **Dynamic single-device decode:** actual B/H/W/device rather than configured global batch/two-GPU allocation. Original constants, conditional saturation and signed offsets are retained; `log1p` is disclosed as mathematically equivalent, not bit-exact.
5. **Actual graph:** batch1/2/4 CPU forward/backward/SGD and gradient audits pass. Expected frozen, dormant and initial gamma-gated tensors are enumerated rather than misreported as unexplained missing gradients.

### Deliberately unresolved / retained

- The real production dataset has **not** been migrated to canonical identity metadata and controlled rejection handling. Cross-dataset same-object aliases and collision rates remain unknown.
- Finite radial-boundary regression weight **64** is retained and explicitly regression-tested. Replacing this weighting is a separate corrected-recipe decision, not silently bundled into a numerical guard.
- Minimum10px inference size, corner overflow, grid constants and quantization/saturation heuristics remain unchanged in B0/B1. Their empirical effects are not solved by target tests.
- Full optimizer/scheduler/augmentation/BN/accumulation/resume fidelity and real training-mask frequencies remain unvalidated.
- **168 CPU formula comparisons passed:** default HiFT-shaped MHA maximum absolute difference0, supplementary SDPA FP32 maximum1.49e-7, normalization FP32 maximum1.91e-6. Official v1.6 source was hashed and retained as inert text. This is formula agreement on modern kernels, **not actual torch1.6 execution, historical CUDA parity, or complete-model equivalence**.

Consequently G1/full M1 remains **PARTIAL**, despite the passing corrected-API test scope.

## C. Authentic data and matched B0/B1 results

### Provenance and protocol

[Acquisition audit](wave2_data.md) and [procedure](../docs/wave2_data.md) document all four source investigations. Publisher-linked [KAUST UAV resources](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset) expose valid HTTP206 ranges for the original archives. ZIP64 directory retrieval enabled selective original members rather than4.69GB/14.05GB full downloads.

- Pilot fixed **before inference** by complete-sequence size and target-category diversity: `building5`161 frames, `uav6`37, `wakeboard7`67. These are complete official10fps segments, not synthetic or locally downsampled clips.
- Every JPEG fully decoded; official six-digit ID sets/ranges, annotation rows, finite nonabsent xywh, initial box, original member CRC and per-file SHA256 validated. Prepared265 images and3 annotation files independently verified byte-identical to original members.
- Script payload total including probes/directories/metadata/failed attempts/attributes: **40,675,305 bytes (~38.79MiB)**, below512MiB. Successful pilot transfer alone18,124,403bytes. Full archive SHA256 remains unknown because no full archive was downloaded.
- Official `uav6` frame000024 contains full-NaN GT; README defines it as absent/fully occluded/outside image. The frame/row is retained. All sequences use **count_as_failure**, with no denominator removal, interpolation or replacement selection.
- Origin is **not explicitly established by the publisher**. Primary reproduction preserves literal upstream numbers: initialize raw firstGT plus`[.5,.5,0,0]`, no subtraction of1; evaluate against native GT. Operational tracker coordinates and annotation origin are not falsely equated. Physical support diagnostics separately assume origin1 and report0-vs-1 sensitivity.
- GT-free inference files contain only frame order, initialization and explicit conventions. Later GT enters only CPU evaluation/retrospective analysis, never the tracker.
- Standard primary metrics: continuous xywh IoU; strict success `IoU > {0,.05,...,1}` averaged over21 ordinates; precision`<=20px`; equal-sequence macro average; submitted initialization included. Normalized precision is supplemental only.

### Measured pilot results

Same official checkpoint SHA256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`, original upstream config/layout/crops/penalties/clipping, seed0, FP32/TF32off, local software/GPU and ordered frames. B1 applies sigmoid to cls2 only. Baseline order alternates across sequences; runs are serialized. No retuning.

| Sequence | Frames | B0 success AUC | B1 success AUC | B0/B1 precision@20px | B1−B0 AUC |
|---|---:|---:|---:|---:|---:|
| building5 |161|0.821650399|0.815439219|1 /1|−0.006211180|
| uav6 |37|0.055341055|0.055341055|0.081081081 /0.081081081|0|
| wakeboard7 |67|0.779673063|0.783226724|1 /1|+0.003553660|
| **Sequence macro** |**265**|**0.552221506**|**0.551335666**|**0.693693694 /0.693693694**|**−0.000885840**|

**No aggregate improvement.** Sigmoid fixes the score interpretation/range, not guaranteed accuracy or calibration. The mixed result is retained, including deterioration.

All530 submitted predictions are finite positive-size boxes; no runner failure or fallback. Excluding initialization, both have33 measured zero-overlap `uav6` frames plus1 unavailable-GT row counted as protocol failure; neither other sequence has zero overlap. Absent-row protocol failure is not misdescribed as observed localization error. All262 tracking predictions differ between baselines, even where thresholded metrics tie.

### Frame-level mechanism and differences

- First divergence is frame000002 in all three sequences, with **identical prior state and exactly identical raw model responses**. On building5 both select cell60, but interpolation LR changes0.70550→0.29379: smoothing alone changes state. On uav6 selection82→71, and wakeboard7 selection49→60: direct ranking also changes.
- Actual selected cells differ on89/160 building5,30/36 uav6 and50/66 wakeboard7 tracking frames. Later differences are stateful trajectories, **not isolated same-input ranking tests**. Both counterfactual policies are additionally reconstructed on each individual response/state.
- B1 has higher/lower frame IoU on64/96 building5,1/1 uav6 (remaining zero-overlap ties),33/33 wakeboard7. Initialization ties are excluded.
- B0 selected scores lie outside[0,1] on247/262 tracking frames; uav6 has2 negative interpolation learning rates, minimum−0.05482. B1 has no selected-score or LR range violations. All observed candidate boxes have positive size and finite policy maps in this pilot.
- [Original matched run](../artifacts/runs/WAVE2-REAL-001/comparison.json), [CPU-recomputed audit](../artifacts/audits/WAVE2-EVIDENCE-001/evidence.json), raw grids and framewise comparisons preserve evidence. Audit rejects changed checkpoint/upstream, frame order/content, critical inference sources, prediction bytes, or inconsistent recomputed metrics.

### Runtime and limitations

Decode-inclusive mean tracking latencies, B0/B1: building5 **11.54/11.44ms**, uav6 **11.41/12.75ms**, wakeboard7 **12.24/11.99ms**. These exclude initialization and post-timing observer/output writes; separate wall-loop rates include those costs. Tracking-only means8.69/8.71,10.45/11.75,9.19/8.97ms. Peak allocated74,036,736bytes and reserved90,177,536bytes for each run. Desktop sharing, short sequences and sequential execution mean **no speed comparison claim**.

This validates the real pipeline, not published full-benchmark reproduction. Only3/123 deliberately short sequences; no confidence/causal claim from correlated frames. Full UAV123/UAV20L and DTB70 scores remain absent. DTB official annotations/config were obtained, but its inspected image share is expired/filtered. UAV variants share source videos and are not independent evidence. Released config names VID/COCO/GOT/YOUTUBEBB, but checkpoint-specific training/pretraining membership and leakage are not independently certified.

## D. Local training feasibility

Primary [training report](wave2_training.md) / [immutable revised CUDA manifest](../artifacts/profiles/wave2-cuda-20261023-b/manifest.json). RTX5070Ti, driver591.74,16,303MiB nominal VRAM; torch2.10.0+cu128/CUDA12.8; Python3.13, FP32/TF32off, deterministic settings, one process/GPU, batches serialized. Seed20261023; random initialization explicitly **not** the official checkpoint.

Actual pinned backbone/correlations/transformer/heads with explicit B2 corrections;127/287 resident synthetic crops; train-mode dropout/BN except frozen layers1–2. SGD momentum0.9/weight decay1e-4, backbone/grader LR5e-4/5e-3, clip10. Two warmups plus five measured full steps each; separate positive/negative backward probes. Batch1 alternates positives/negatives; larger batches50/50—not the released5% negative recipe.

| Batch | Model forward stages median sum* (ms) | Decode/loss median (ms) | Backward median (ms) | SGD median (ms) | Whole step mean (ms) | Synthetic pairs/s | Current alloc/reserved (MiB) | Peak alloc/reserved (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|7.87|4.94|9.48|1.39|25.64|39.00|142.37 /238|176.10 /238|
|2|8.05|5.21|10.57|1.25|26.90|74.34|142.89 /270|191.22 /270|
|4|10.07|5.48|11.62|1.36|31.20|128.22|145.54 /388|307.22 /388|

*Sum of separately measured template backbone, search backbone and combined correlation/transformer/heads medians; not a separately timed forward-total median. Whole step also includes zero-grad, clipping, synchronization, Python and validation overhead. **All fit without OOM**; maximum batch/long-run memory were not tested. Allocator values exclude driver/context/desktop residency; the global start snapshot was4065MiB used.

Finite, nonzero active gradients were checked. Four genuinely dormant encoder tensors, four gamma-gated initial zero tensors, frozen tensors and expected negative-only localization zeros are documented. No fabricated training loss convergence or B2 accuracy claim.

**Planned arithmetic, not measured training:** microbatch4 is the most efficient tested candidate. Effective16 would require4 accumulations;1000 effective updates would process16,000 pairs. Linear resident-graph proxy≈124.79s using measured mean step cost; this is not an input-pipeline or wall-time bound. Accumulation has not been implemented/timed and changes BN/dropout and per-class/per-mask loss normalization. Full data decode/augmentation/loader/H2D, validation/checkpoint overhead and schedule/resume remain unmeasured. No70-epoch duration or source-batch220 equivalence is asserted.

## E. R1 diagnostic preparation and evidence gaps

[Retrospective diagnostics](../src/hift_research/diagnostics/source_support.py) record GT original width/height/area/aspect, motion and scale change, actual search side/origin, resize scale, source-pixels/model-pixel ratio, nominal resized footprint, crop/image/joint containment, geometric contained area and resized contained area, IoU/center error/normalized center error, and explicit unavailable annotations. These are **geometry proxies, not entropy, optical resolution or recovered visual detail**. Enlargement creates no new information.

Original [sequence attributes](../artifacts/acquisition/WAVE2-UAV10-ATTRIBUTES-002/attributes.json) were also selectively obtained with CRC/SHA provenance. uav6 is officially flagged SV/LR/FM/FOC/POC/CM/SOB; wakeboard7 SV/ARC/VC; building5 CM. These flags are **sequence-level**, not per-frame event labels. No separate official blur flag or measured per-frame distractor confusion was established. Absent GT is not relabeled as a specific occlusion cause.

### Real exploratory observation

At uav6 frame000004, GT is only **9×8 original pixels**, while its nominal resized footprint is **57.4×51.0 pixels**. It is still fully inside the recorded crop, yet both trackers have zero IoU. GT center moved18.36 source pixels since the previous10fps frame—about2.3 target heights—so fast motion, window/selection behavior, source detail and model suitability are confounded.

There are25 valid tracking frames with original minimum side<16px but nominal resized minimum side≥32px. Twenty-three have zero IoU, **but22 are already outside the search crop**. Conditioning on full crop containment leaves only3 such frames, with1 failure. Building5 and wakeboard7 have no comparable small-source/large-footprint contained stratum. Thus the pooled association is dominated by sequence/time/motion/crop-drift confounding, not proof of a resolution-fusion deficiency. Larger nominal footprints after drift can correspond to **zero actual target area delivered to the crop**.

The assumed origin1-vs0 changes no crop-containment conclusions in this pilot, but maximum per-frame valid IoU differences (including initialization) reach0.107/0.283/0.182 for building5/uav6/wakeboard7. Origin uncertainty is not waved away for tiny objects; primary native metrics are unchanged. Prespecified source-area bins and explicitly exploratory within-sequence correlations are stored without frame-iid significance tests or causal claims.

**Still needed:** broader contained low-support controls, reliable origin evidence/sensitivity, per-frame motion/occlusion/blur/confusion labels where legitimately available, source-detail measurements beyond bbox geometry, frozen B2, leakage-safe evaluation, preregistered falsifiable mechanism, and comparison against the closest crop/resolution/fusion literature. No R1 novelty or training justification follows from this pilot.

## F. Recommendations and next bounded experiment

1. **M1:** accept the independently tested corrected APIs and graph only; retain overall PARTIAL until a real-data identity adapter, rejection accounting, retained weight-policy decision and complete training lifecycle are reviewed.
2. **M2:** accept acquisition and matched pilot as genuine measured progress; do not accept full benchmark reproduction or an accuracy improvement. Preserve B0 as reference; do not silently promote B1 as universally better.
3. **B2:** hardware supports a small synthetic microbatch4 pilot, but a real training run is **not ready**. Freeze a legal video-disjoint train manifest, canonical identities, augmentation/target policy, optimizer/scheduler/BN/accumulation reductions and resume invariants first. Benchmark sequences are not to be repurposed as training data.
4. **R1:** HOLD architecture/training. Existing literature challenges generic fusion/crop novelty; present geometric evidence does not identify a unique mechanism or establish novelty.
5. **Next bounded experiment:** CPU-only real-training-data contract preflight on an independently authorized small training manifest: sample1024 pairs with no SGD, log canonical identities/rejections, negative prevalence and target/mask/support distributions, enforce no benchmark-video overlap, and verify optimizer/scheduler/BN/resume semantics separately with synthetic fixtures. Proposed≤10min after assets/recipe approval; not run this wave. Only then consider a separately authorized short corrected-B2 pilot. Continue no-training source-support diagnostics without feeding later GT into a deployed tracker.

### Reuse without restarting completed work

Reuse existing immutable weights/data/runs. Commands for independent verification:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
git -C third_party/HiFT status --porcelain
git status --short
# New audit destinations only; never overwrite WAVE2 evidence directories.
.venv/Scripts/python.exe scripts/verify_recorded_geometry.py --runs artifacts/runs/WAVE2-REAL-001 --output artifacts/audits/NEW-GEOMETRY.json
```

Full acquisition, matched-run, profile and audit commands are preserved in their scripts/manifests and subsystem reports. Do not rerun downloads or replace results merely to obtain a more favorable score.
