# Wave4 — serial production lifecycle and bounded pilot contract

## Boundary and present assessment

**No real image entered SGD, no pretrained tensor was optimized, no official tracker fine-tuning, no R1, and no full training.** This worker used exactly **16 synthetic CUDA effective updates**: eight in the supplemental lifecycle proof, four in the initial shared-controller proof, and four newly authorized authentic-layout regression updates. The cap was explicitly extended12→16 before the new regression. All three exclusive GPU tokens were released, jobs collected successfully, and no retries or additional GPU work occurred. All synthetic models were initialized entirely randomly.

- **W4-G4 lifecycle:** PASS for the serial/no-prefetch atomic-boundary fallback: authentic Windows inputs, repaired real-forward/fullDEV path, and authentic-layout synthetic CUDA replay all validated. The initial real storage-layout failure and old fixtures are preserved as failed/limited evidence, not relabelled passes. Asynchronous loader-queue or multiworker exact resume is not claimed.
- **W4-G5 pilot recommendation:** **conditional GO for a future fresh bounded grader-only pilot**,100updates/1600training attempts/**900seconds total**. This is **not authorization for real SGD**, a promised scientific signal, an architecture/convergence result or full-training readiness. Manual resumption requires separate human resource review and remaining-time enforcement.
- Reference70-epoch/four-dataset reproduction remains unvalidated; a100-update pilot cannot stand in for it.

## One production controller, two strictly separated permissions

The future [pilot command](<F:/Code/HiFT/scripts/run_b2_pilot.py>) calls [future authorization/dispatch](<F:/Code/HiFT/src/hift_research/training/wave4_pilot.py>), which imports the **same** [SerialPilotController](<F:/Code/HiFT/src/hift_research/training/wave4_pilot_engine.py>) tested by [production verification](<F:/Code/HiFT/scripts/verify_wave4_production.py>). Synthetic mode requires the exact indexed synthetic Dataset class and bounded updates. Real mode requires a separate future human grant, exact config/data/policy/budget/device binding, acknowledgement flag and exclusive GPU token. Real arrays are never relabelled synthetic to bypass a guard.

[Indexed loader](<F:/Code/HiFT/src/hift_research/training/wave4_loader.py>) consumes the source-audit tuple API `(record,arrays-or-None)` and the unchanged reject-aware collator. Four **attempt** slots form each microbatch;16attempt slots form one effective window. Rejections reduce actual accepted batch size; they never trigger refill, replacement, hidden retries, or changed negative frequency. An all-rejected window commits its rejection ledger and performs **no** optimizer update. Loss denominators use actual accepted global class counts, BCE cells, and IoU-weight sum with one epsilon; clip/step once per nonempty window. This is not physical220 equivalence, nor BN/dropout equivalence to a physical16 batch.

A dedicated CPU `torch.Generator` isolates DataLoader iterator base-seed generation from the parent model/dropout RNG, including num_workers=0. Candidate seeds are explicit index-derived values. Missing/decode/unexpected exceptions stop the pilot with the failing index. Diagnostic fault receipts are a separate explicit mode, not production fault skipping.

### Exact safe recovery contract

The controller's cursor advances only after a complete successful effective update or an explicitly all-rejected window. It cannot save or continue a tainted/uncommitted window. A midwindow interruption may already have changed gradients, dropout RNG and BN buffers; discard that instance, restore the **previous atomic boundary**, then recompute the entire same indexed window. This is replay of the same attempts, not replacement or a new distribution. Repeated I/O must be reported separately; there is no automatic retry.

Boundary checkpoints contain strict model/BN buffers, optimizer momentum/groups/options, source-valued scheduler, epoch/update/window/committed cursor, semantic attempt/loss ledger, clipped gradients, Python/NumPy/Torch CPU and used CUDA RNG, config/data/source/device identities, and future dev/stop-control auxiliary state. All tensors serialize on CPU and load using **weights_only=True**; restoring requires the exact previously recorded file SHA256. Source, config, data/policy/reservations, device, model key/shape/dtype/finite state, optimizer options/ordered IDs/momentum coverage, gradients, and cursor clocks are checked. Checkpoints publish completed fsynced bytes atomically using a no-overwrite hardlink; existing runs/checkpoints are never overwritten.

## Actual layout failure and explicit W4 repair

Lead's first authentic forward-only run failed at immutable source cross-correlation `view()`: NumPy HWC→CHW transpose plus the unchanged collator yielded channels-last-strided NCHW tensors. Correct shape was insufficient. No backward/SGD occurred. Failed001 and its source snapshot remain intact.

New [layout bridge](<F:/Code/HiFT/src/hift_research/training/wave4_layout.py>) validates CPU FP32 strided [N,3,127,127]/[N,3,287,287] and applies `.contiguous(memory_format=torch.contiguous_format)` **before H2D** inside the actual shared attempt-buffer path. It proves logical pixels unchanged; targets remain untouched. There is no channel permutation, resize, range/dtype conversion, architecture change, or frozen source/policy edit. Wrong shape/dtype/device/non-strided/nonfinite input fails closed. CPU tests reproduce the old view failure on the real-shaped NumPy transpose/stack fixture, then verify C-NCHW strides, pixel identity and boundary recovery. Lead's repaired real profiling and future CLI use this same bridge, not a profiler-only workaround.

## Primary shared-production CUDA evidence

Immutable [layout production manifest](<F:/Code/HiFT/artifacts/wave4-lifecycle/production-layout-cuda-001/manifest.json>) and [verification](<F:/Code/HiFT/artifacts/wave4-lifecycle/production-layout-cuda-001/verification.json>) use **noncontiguous synthetic NumPy image arrays** matching the actual storage path and record **13/13 checks true**:

- uninterrupted2updates versus interrupted1update + midwindow crash (**zero** extra SGD) + restored1update;
- parameters/BN buffers, optimizer/momentum, scheduler/LR, clipped gradients, Python/NumPy/CPU/CUDA RNG, committed cursor/window/update and timing-free semantic sample/loss trace are bit-exact;
- crash leaves parameters/SGD/cursor uncommitted, tainted continuation and checkpointing are rejected, and no tainted checkpoint appears;
- exactly **four** effective CUDA updates, physical4/nominal effective16, source epoch0/head-only, seed20261023, RTX5070Ti; every active gradient finite, initial gamma-gated and dormant encoder `eles` exceptions declared.

Total correctness harness runtime **9.398s**. Atomic checkpoint serialization **0.45157s**, strict restore **0.51483s**. [Boundary checkpoint](<F:/Code/HiFT/artifacts/wave4-lifecycle/production-layout-cuda-001/update1-boundary.pt>) SHA256 `756922f1e1abb92b652f96da0ac69a7d0e51ebeb74db7158e043a0af3611a5f6`.

| Synthetic production-controller stage | Cold update1 | Update2 |
|---|---:|---:|
|Indexed synthetic inputs/collation/layout/window buffer|0.14380s|0.11021s|
|H2D|0.01761s|0.00744s|
|Forward + corrected loss|0.87772s|0.16796s|
|Backward|0.25827s|0.11117s|
|Gradient audit + clip|0.12809s|0.05875s|
|SGD|0.02471s|0.00834s|
|Total window|1.45563s|0.46698s|

Post-return allocated memory165.154→163.464MiB; reserved282→316MiB; cumulative peak allocated230.950→256.037MiB. Restored second update0.46069s, allocated162.796MiB and peak254.833MiB. These are allocator observations over **two** updates, not proof of long-run absence of leaks. Volatile stage/memory observations are separate from exact semantic comparisons. Models were discarded between paths; process allocator scope and cold-start effects are disclosed.

**These synthetic-input timings do not decode/crop/augment real images and alone are NOT a full-input-path pilot budget.** Old Wave2 resident-crop rates are not used. Lead's separate actual [REAL-FORWARD002](<F:/Code/HiFT/artifacts/wave4-feasibility/REAL-FORWARD-002/manifest.json>) and [BUDGET001](<F:/Code/HiFT/artifacts/wave4-feasibility/BUDGET-001.json>) now establish the following bounded engineering projection.

### Measured actual path and total-wall budget

Exactly the first32authentic training candidates were processed:28positives/4safe negatives, no outcome-dependent selection. Actual read/hash/decode/crop/augmentation/target input cost averaged75.568ms/attempt; raw real no-grad forward+loss averaged40.7705ms/physical4 batch. The same future `paired_development_loss` path processed the entire fixed128weighted control cohort in5.89058s, before training-mode forward exposure, with source backbone/random grader. Its initial unoptimized paired loss3.453117579 is a readiness observation, **not training progress, tracking performance or a substitute for a future freshly computed update0 baseline**. Parameters were SHA-identical, no `.grad` appeared, no optimizer/backward/SGD occurred, and disposable grader-BN adaptations from later train-mode forwards were discarded with the model. Real forward peak allocated138.718MiB/reserved194MiB; observed host peak2071MiB. Allocator figures exclude full driver/context VRAM and do not establish long-run stability.

Budget central estimate **192.018s≈3.20min**, engineering range **144.013..573.143s≈2.40..9.55min**, hard cap **900s**. It includes1600input attempts, collation/layout/H2D, per-batch forward using the maximum of actual no-grad cost and warm synthetic train/autograd floor, new-layout synthetic backward/audit/clip/SGD, four full128DEV checks (23.562s), four checkpoints and cold fixed overhead. The backward estimate has **n=1warm effective window**, not real-image gradients. This is an engineering range, not a confidence interval, measured complete training throughput or convergence forecast. No overlapping-loader speedup is assumed; cache locality, contention and storage may change cost.

**900s is a cumulative pilot cap, not a fresh allowance per resume.** A committed boundary cannot automatically know time spent after it in a failed window or checkpoint operation. The current GO recommends a fresh invocation only. Before any manual resume, a human must review grant expiry/token and all previous invocation, failed/uncommitted window, checkpoint save/load and I/O time, deduct it from900s, and enforce the actual remaining wall externally. If that cannot be reconstructed/enforced, manual resume is NO-GO. No automatic retry is allowed.

The earlier [contiguous-fixture production manifest](<F:/Code/HiFT/artifacts/wave4-lifecycle/production-cuda-001/manifest.json>) and [verification](<F:/Code/HiFT/artifacts/wave4-lifecycle/production-cuda-001/verification.json>) had13flags true in7.641s, but did not exercise authentic NumPy storage layout. That historical proof was preserved. The four new updates test a newly discovered real-input correctness defect, not a repeat of a completed benchmark.

### Supplemental epoch0/10 device proof

Earlier immutable [CUDA manifest](<F:/Code/HiFT/artifacts/wave4-lifecycle/cuda-001/manifest.json>) and [verification](<F:/Code/HiFT/artifacts/wave4-lifecycle/cuda-001/verification.json>) tested the [synthetic-only lifecycle](<F:/Code/HiFT/src/hift_research/training/wave4_lifecycle.py>) at source epochs0 and10:18/18 comparison flags true, eight total effective SGD updates, physical4/effective16,18.605s total. It saves pending CPU batches/global denominators and accumulated gradients midwindow, with strict CUDA RNG restore. Per-stage/memory metrics were **not** instrumented in that run; it was not restarted to obtain them. It remains supplemental, not a substitute for the shared production-controller boundary proof.

[Initial CPU manifest](<F:/Code/HiFT/artifacts/wave4-lifecycle/cpu-001/manifest.json>) had18/18flags true at epochs0/10 with physical1/effective2. It predates the shared controller and is historical, not the final data-bound configuration.

## CPU/Windows loader evidence

[Tests](<F:/Code/HiFT/tests/test_wave4_lifecycle.py>) first passed11CPU tests, including an actual Windows spawn2 DataLoader, serial/spawn identical indexed arrays/order/seeds, accepted/rejected/all-rejected receipt behavior, missing/decode/unexpected exceptions, and explicit cursor reconstruction. After the shared production controller was added, **14CPU tests passed in38.32s**, including atomic boundary crash/replay, tainted-instance rejection, no-refill all-rejected windows, and future real-authorization denial before output/CUDA initialization. A pre-run missing-parenthesis syntax error in the new controller was fixed and tests rerun; it consumed no CUDA updates.

Final local CPU regression suite: **25passed in46.29s**. The final partial-window test uses18attempts/physical4/window16, observes16then2accepted, BCE denominators1936then242, and exactly two zero-grad/clip/SGD calls. An independent same-physical2 tail calculation has bit-exact loss and clipped gradients with restored dropout RNG; it makes no physical16 BN-equivalence or fictitious4×division claim. This includes authentic002 four-slot serial/spawn2 bit-exact transport, cursor reconstruction and injected I/O failure/recovery **without changing any source media**, plus the new authentic-shaped layout failure/repair, malformed layout contracts, and noncontiguous-input shared-controller replay. Lead separately owns the larger16-index authentic Windows order/spawn/fault probe, which passed6flags. Neither transport probe performs real-image optimization.

## Reference versus preregistered pilot

[Reference config](<F:/Code/HiFT/configs/B2-Reference-W4.json>) preserves the actual source reference:70epochs, global physical220/two devices, VID/COCO/GOT/YOUTUBEBB mix, source-valued LR table. That mix is not validated locally. Corrected graph/global normalization, stable optimizer groups preserving grader momentum at source epoch10, and explicit frozen layer1/2 BN are disclosed deviations, not a released-training reproduction claim.

[Pilot config](<F:/Code/HiFT/configs/B2-Pilot-W4.json>) instead fixes:

- **Grader-only100effective updates** means the **entire corrected grader: fusion, transformer and prediction heads**, not only terminal convs. The entire backbone and backbone BN remain frozen. At most1600 unique training attempt slots/100windows; nominal physical4/effective16, accepted counts may be smaller. If all-rejected windows prevent100updates within the attempt budget, completion fails rather than refilling.
- **Actual source epoch clock**,600000attempts/epoch, so this bounded pilot stays at epoch0. Grader LR is **0.0025**, not the optimizer's nominal BASE_LR0.005: source warmup normalizes by the normal curve's0.01 start. Backbone ratio0.1 gives0.00025 but all backbone weights and BN stay frozen. No compressed warmup/unfreeze.
- SGD momentum0.9, weight decay0.0001, global clip10, stable group membership. Seed20261023 is an explicit deviation from source CLI default123456.
- Planned initialization: strictly loaded source AlexNet backbone with the approved **five-counter-only zero initialization** and random corrected grader; never official general_model checkpoint fine-tuning. This is **not** how synthetic SGD was initialized: every synthetic optimized tensor was random.
- TRAIN MOT17-02/09/10 versus DEVELOPMENT MOT17-05, roles chosen from primary source lineage before pair-loss outcomes; not the15frozen UAV evaluation sequences. DPM/FRCNN/SDP aliases are not independent videos.
- Temporal positives share a source trajectory fragment at nonzero native-frame gap1..19. Negatives are eligible simultaneously present distinct source fragments with full A-source-support exclusion, not arbitrary cross-video/global-person identity assumptions. This conditioned negative sampler is an explicit source deviation. Candidate negative prior0.05 and zero retry/replacement remain fixed.
- DEVELOPMENT is a predeclared **64positive/64known-safe-negative** control cohort:128attempts, alternating positive/negative indices, seed20261024, no shift/scale/blur/color/flip. It is not the training Bernoulli distribution or a prevalence estimate. The64negative draws represent38unique source pairs/array sets with26repeats (maximum multiplicity4), while all64positive pairs are unique; this is **not128independent observations**. Repeated eligible draws are not rejection replacement. The fixed weighted control cohort is not rebalanced after observing this dependence. Missing controls fail readiness; never redraw a favorable cohort.
- Paired development loss at effective updates0/25/50/100, **not tracking AUC**. Primary is the fixed-last100 checkpoint; lowest-dev/earliest-tie checkpoint is secondary and must be reported separately. No15-sequence GT tuning.

Operational reliability success (finite losses/gradients,100updates within exact budget, valid checkpoint/replay, no hidden replacements/identity drift/OOM, fixed-last dev loss<=3×baseline) is separate from **preliminary scientific signal**: preregistered fixed-last dev loss must be **<=0.95×initial**. Otherwise report **NO promising signal**, even if reliability passes; best-dev cannot rescue the fixed-last criterion. Neither result establishes tracking accuracy, architecture benefit, convergence or70-epoch training success.

Resource accounting includes1600training attempt slots **plus up to512development pair forward evaluations** of the fixed128attempt weighted cohort (38unique negative pairs,26negative repeats) at four checkpoints, checkpoint I/O, actual read/decode/crop/augmentation/targets/collation/H2D and model execution. The measured total hard wall cap is900s. Stops include nonfinite state, input/hash drift, OOM/device error, reserved memory>80%, deadline, exhausted attempts before100updates, or two consecutive dev losses>3×baseline. No automatic continuation after failure.

## Data/config binding and snapshot separation

Authoritative source [temporal manifest](<F:/Code/HiFT/artifacts/wave4-source/SOURCE-001/temporal_manifest.json>) SHA256 `2660a96d72ed1645b822120d81835f7a66f93b65da9a4648f6a16fcc04bdba54`. Current [TRAIN002 policy](<F:/Code/HiFT/artifacts/wave4-preflight/FROZEN-002/train_policy.json>) SHA256 `c0d811b945f385084daa8ed27042e2bc0178a4859ffab34868fda8304d33afa9`; [development002 policy](<F:/Code/HiFT/artifacts/wave4-preflight/FROZEN-002/development_policy.json>) SHA256 `15bf57bc1b7107cf5d6ca51ed72a54c520e13fda7080a2014c833bb2afcd6a36`. Config now binds these002 source/TRAIN/development identities and reservations SHA256 `a4f0b91e0c94181f4f8dbc09a09762fa722d97cead728bd553c333447f508e7c`. The fixed [DEV ledger](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-002/development/candidate_ledger.jsonl>) SHA256 is `eae252a734198935c62a80fdbd8239d05169c710a7182d6188de2d8d21c1efeb`:128/128accepted,64+64. Full TRAIN1024 yielded973accepted positives,50accepted safe negatives and one explicit actual-footprint negative rejection; zero hidden refill. Owner's full replay/archive audits passed. Repaired real-runtime002/fullDEV and BUDGET001 evidence are now bound as metadata; budget.max_wall_seconds=900 and readiness.recommendation=GO_RECOMMENDATION_NOT_AUTHORIZATION. Both current-wave and production-training authorization remain false; no grant file exists. The first preflight001 encountered a NumPy-bool JSON serialization failure; its partial evidence and source snapshots were preserved and a new002 policy/run created, not overwritten. Do not reuse obsolete001 policies after code changes.

The latest authentic-layout CUDA proof executed config SHA256 `836875f27d5d1d36cbc222f9d8c036da292b06cdcf5959facab528906b7a2f0f`:002data/DEV identities were bound in metadata, wall budget was null, and **all optimized arrays were still exclusively synthetic**. Initial contiguous production proof used `3fdf5911b270221af8e862e43b57e1487cf98768a57cc889c7913ca9e878ebe8`; earlier supplemental CUDA proof used `999912257ff5f7353ca290e1d27beb6e9568394ffdeb218d41e4ecc5a90d8493`. Those two earlier snapshots had unbound real-data metadata. Every snapshot is preserved unchanged. Later readiness/wall-budget metadata does **not** mean an earlier CUDA run used the final config. [Final metadata binding audit](<F:/Code/HiFT/artifacts/wave4-lifecycle/final-config-binding-001.json>) records final config SHA256 **`420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e`**. All29critical executing Python files are byte-identical to the latest layout proof; the final differences are allowed readiness/evidence and900s wall-budget metadata. The audit validates every bound evidence SHA and authentic Dataset identity, then proves GO+ack still raises PermissionError without a separate future grant. No further GPU run occurred.

Source assignment HEAD `28e9e197c62a21450587465e4c32abdf97fb88d3`; immutable upstream `992aa7c1605360d607af84cb804b93cd8ee81caf`. Latest layout-proof source-tree SHA256 `c4662e3520630d2254692f0ce812882ee7e224059366edae4317ac53e7cf9273` (initial contiguous proof `9d4ba701e39bac982692115eea9e31780317da9ecaa7adcc6ddedeb59cc136f0`). Empty tracked diff alone does not establish a clean worktree: new untracked worker files are included in the provenance tree and exact executing snapshots. This worker made no commits; lead owns integrated final tests/commits/ending cleanliness.

## Future command — NOT RUN, NOT AUTHORIZED

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_b2_pilot.py \
  --config configs/B2-Pilot-W4.json \
  --authorization-file <separate-future-human-approved.json> --ack-real-sgd \
  --output artifacts/b2-pilot/<new-exclusive-run-id>
```

The external future grant must bind the exact config/source-data/policy hashes,100updates/1600training-attempt and measured wall budgets, device, expiry, human request reference and exclusive token. **No such authorization file was created.** The current wave's real-forward-only permission and any future GO recommendation are not real-SGD permission. The reference70-epoch command is not recommended or launched.
