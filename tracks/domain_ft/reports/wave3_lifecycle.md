# Wave3 — source-derived B2 lifecycle, synthetic CPU verification

**Boundary:** bounded synthetic optimization only; no real image entered SGD, no GPU, no B2 full training/fine-tuning, no official tracker checkpoint training, no R1 work. Existing graph/profiler/upstream/old reports/configs/registry untouched. New proposed recipe is **fail-closed**, not a training authorization.

## Gate assessment

- **Synthetic lifecycle sub-gate: PASS (bounded CPU evidence).** Source-valued schedule, freeze/BN policy, correct effective-window normalization, optimizer stepping, restricted mid-accumulation resume, and minimal legacy-backbone counter compatibility are tested.
- **W3-G2: PARTIAL.** Synthetic correctness does not freeze/accept a real-data distribution or validate production multiworker replay. Canonical manifest binding and real recipe acceptance are separate conditions.
- **W3-G5: PARTIAL / no launch.** Authentic training access, canonical identities/reservations, accepted negative distribution, full input pipeline cost, real accumulation/BN policy, and a separately authorized bounded pilot still need acceptance. A preflight-only data manifest cannot grant production training. GPU availability does not change this.

Implementation: [lifecycle](<F:/Code/HiFT/src/hift_research/training/lifecycle.py>), [recipe](<F:/Code/HiFT/src/hift_research/training/recipe.py>), [verification CLI](<F:/Code/HiFT/scripts/verify_training_lifecycle.py>), [tests](<F:/Code/HiFT/tests/test_training_lifecycle.py>), [proposed config](<F:/Code/HiFT/configs/B2-wave3-proposed.json>).

## Actual source audit, not generic defaults

Authoritative immutable sources: [trainer](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L75-L104>), [epoch loop](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L117-L173>), [scheduler](<F:/Code/HiFT/third_party/HiFT/pysot/utils/lr_scheduler.py>), [core defaults](<F:/Code/HiFT/third_party/HiFT/pysot/core/config.py>), [released YAML](<F:/Code/HiFT/third_party/HiFT/experiments/config.yaml>), and [checkpoint helper](<F:/Code/HiFT/third_party/HiFT/pysot/utils/model_load.py>). Upstream SHA `992aa7c1605360d607af84cb804b93cd8ee81caf`; assignment HEAD `fba88e5b3452bf6e92980f14a52604d5d93eb0b7`. The recipe reads literal defaults and recursively merges actual YAML; exact tests execute the source scheduler and optimizer-builder AST bodies without a trainer/GPU launch.

| Item | Actual merged source | Proposed treatment / deviation |
|---|---|---|
|Epochs / data|70;600,000 candidate pairs/epoch; VID/COCO/GOT/YOUTUBEBB|Source settings preserved in proposal, not a claim these datasets are validated locally|
|Batch / devices|Global220; NUM_GPU2; DataParallel; workers6|One process, microbatch4×55 proposed; no real run or multi-GPU implementation|
|SGD|BASE_LR0.005; backbone ratio0.1; momentum0.9; WD0.0001; clip norm10|Exact numerical defaults; clip/step once per effective window|
|Epoch0–9|All backbone parameters frozen; all backbone BN eval; grader train|Explicitly applied after model.train, not inherited accidentally|
|Epoch10+|Enable layer3/4/5 weights and BN; rebuild optimizer/scheduler|Stable groups retain grader momentum; layer1/2 weights and BN always frozen|
|Backbone initialization|alexnet-bn.pth; model-load helper uses non-strict model loading|Restricted weights-only; strict normalized keys with ONLY five approved counter defaults|
|Grader initialization|Random, source CLI seed123456; channel-attention gamma0|Synthetic verification seed20261023 disclosed; official released tracker checkpoint not substituted|
|Negative candidate probability|0.05, not core default0.2|Preserved; attempted/accepted/rejected candidates must remain distinct; no replacements|
|Target boundary weights|Retained source behavior can leave64|Not clamped/retuned; weighted denominator audit preserves64|

### LR normalization trap and epoch semantics

Source warmup raw values are `0.005*2**(i/5), i=0..4`; normal raw curve is65 logspace values0.01→0.0001. `WarmUPScheduler.start_lr` is normal[0]=0.01, and each group receives `raw_lr*initial_lr/0.01`. Therefore **effective grader LR** is:

- epoch0: **0.0025**, not0.005;
- epochs1–4:0.00287174588749,0.00329876977693,0.00378929141628,0.00435275281648;
- epoch5:0.005;
- epoch10:0.0034891529242993317;
- epoch69:0.00005.

Backbone is0.1×these values whenever trainable. Tests compare all70 epochs against the actual scheduler module exactly for grader and within floating tolerance for backbone. Optimizer updates/microsteps do **not** advance this schedule. Epoch changes require completed data epoch and a flushed final partial window; checkpoint records epoch/update/microstep clocks independently.

The source rebuilds SGD exactly at epoch10 and discards existing grader momentum; the test executes this actual function and proves its new state is empty. Proposed fixed groups deliberately preserve momentum. Source fresh entry>=10 can leave layer1/2 BN in train mode, whereas an uninterrupted path first freezes them; explicit BN modes remove this entry-path dependence. Both are disclosed changes, not a claim of exact released training reproduction.

Source dataset length is600,000×70 and trainer floors `num_per_epoch=600000//220=2727`. Source nominal epoch therefore consumes599,940 examples before its next epoch boundary; the proposal instead defines actual data epochs and includes the final60-example partial effective window. This is an explicit epoch-tail policy change; source counters cannot be relabelled as identical.

## Correct accumulation and limitations

The synthetic window buffers **all CPU inputs/targets before any forward**, computes total class/mask denominators, then performs one backward contribution per microbatch. For the whole effective window:

`L = 0.5*sum(NLL_positive)/N_positive + 0.5*sum(NLL_negative)/N_negative + sum(BCE)/N_cells + 2.4*sum(weight*IoU_loss)/(sum(weight)+1e-6)`.

Empty classes contribute differentiable zero, not an extra normalization factor. Allzero localization masks stay zero; raw nonlinear masks are the existing corrected graph adapters. IoU local normalized losses are converted back to weighted numerators using the local denominator, then divided by the global denominator with **one** upstream epsilon. Final partial windows use actual observed counts; clip and SGD occur once, with no phantom divisor or silent tail drop. Tests include unequal positive/negative counts, different microbatch sizes, weights64/1.5/0.0001, and independent loss/gradient comparisons.

Numerical scope: float64 loss/gradient agreement uses rtol=atol1e-12; float32 local-vs-global denominator/reduction rounding uses rtol1e-5/atol1e-6. This is **not bit-exact physical-batch equivalence**. Actual train-mode BatchNorm running statistics and outputs differ with microbatch partition; dropout partitions are not promised equivalent. Source DataParallel independently normalized losses also need not equal a globally normalized220-example objective. Synthetic replay is bit-exact only with the **same microbatch partition and order** in this fixed CPU environment.

No production DataLoader queues, multiprocessing augmentation workers, real rejection/resampling streams, DDP, AMP, cross-device resume, or real accumulation throughput are validated. Buffering a full effective window can add host-memory cost not present in the previous resident-crop profile.

## Restricted checkpoint/resume evidence

Checkpoint contains tensors/primitives only: strict model/buffers, stable SGD state, explicit scheduler table/epoch, epoch/update/microstep counters, Python/NumPy/Torch CPU RNG, synthetic sampler order/position, pending buffered CPU batches/targets and global denominators, pending backward position, accumulated named gradients, trace, config/data identities, and implementation file hashes (model/factory/head/source adapters). No CUDA state is used; restoring a CUDA-used state is rejected rather than falsely claiming support.

Load uses **`torch.load(weights_only=True,map_location='cpu')`**, never weights_only=False or a fallback. It rejects changed source/config/data identity, model keys/shape/dtype/nonfinite values, optimizer group/ordered-ID/LR/options mismatch, momentum shape/dtype/nonfinite state, gradient dtype/shape/nonfinite state, epoch clocks, pending denominators/IDs, sampler/trace/update disagreement, and custom pickled Python objects. On restore failure the caller must discard that instance; no resumed step is accepted.

Tests interrupt at four positions across epochs0,4,5,9,10,69 (24 combinations), including warmup endpoints and mid-accumulation. These tests reconstruct a fresh model/optimizer in-process and reload the on-disk restricted checkpoint; fresh-interpreter/process/worker replay is not claimed. Uninterrupted and interrupted/restored runs agree **bit-exactly** on sample order/loss trace, model parameters/BN buffers, momentum, scheduler/LR, sampler, Python/NumPy/Torch RNG, and final gradients; final singleton windows step once. Separate actual127/287 B2 graph tests prove trainable/frozen modes, dormant encoder `eles`, four initial gamma-gated zero-gradient parameter tensors, gamma activation after updates, and exact mid-window replay. Toy tests isolate schedules/normalization but are not substituted for the actual graph proof.

**Primary immutable evidence:** [CPU-b manifest](<F:/Code/HiFT/artifacts/wave3-lifecycle/cpu-20261023-b/manifest.json>), [verification](<F:/Code/HiFT/artifacts/wave3-lifecycle/cpu-20261023-b/verification.json>), and [restricted mid-accumulation checkpoint](<F:/Code/HiFT/artifacts/wave3-lifecycle/cpu-20261023-b/synthetic-mid-accumulation.pt>). All nine actual-graph comparisons are true, final singleton stepped once, first gamma0 and final gamma0.00006793145439587533. Per-microbatch losses, clipping norms, epoch10 LR and BN modes are recorded. The12.66s total harness runtime includes source/data-hash validation, two actual synthetic paths, checkpoint I/O and the separate backbone audit; it is not a speed benchmark. Python3.13.0/torch2.10.0+cu128/NumPy2.4.4, Windows10 build19044, Intel64 Family6 Model183 Stepping1, one thread, seed20261023, CPU only.

Primary source-tree SHA256 `10a793ee27418d52486dc1716d0196baf0aee1af73ba07dd3bc050249efab074`; dirty-diff SHA256 `4b992507d5a67ddd5f67b372b5d7dfd3bb84dbf305b2f635b9551bd50a4c33be`; exact bound config SHA256 `e5a44406d3b05d67b87e9377bd9ba799c9844b2d606a862cd2509b3f3e593cb4`; checkpoint SHA256 `7aba37a8fd342d757998fad0d55a3781ac91bfeaf38863b7258ad3d3630a7297`. Source-tree identity covers untracked worker files, unlike diff alone. Exact executable source snapshots and per-output hashes are included; later report-only edits are not attributed to the measured source snapshot.

Historical immutable [CPU-a manifest](<F:/Code/HiFT/artifacts/wave3-lifecycle/cpu-20261023-a/manifest.json>) and [verification](<F:/Code/HiFT/artifacts/wave3-lifecycle/cpu-20261023-a/verification.json>) also had all nine comparisons true in4.91s, but predate stricter optimizer/source identity checks and the approved legacy-counter audit. Its `bounded_synthetic_pass` label is a synthetic sub-gate, not all W3-G2 conditions. Neither run was overwritten.

CPU command after hardening:

```bash
.venv/Scripts/python.exe -m pytest tests/test_training_lifecycle.py -q
# 53 passed, 11.19s; CPU synthetic only
.venv/Scripts/python.exe -m pytest tests/test_training_lifecycle.py tests/test_training_profile.py -q
# Final bound-config CPU check: 68 passed, 13.90s; existing graph/profile tests preserved; upstream clean
```

## Minimal backbone compatibility, not partial weight loading

[Source backbone asset](<F:/Code/HiFT/third_party/HiFT/pretrained_models/alexnet-bn.pth>) SHA256 `933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8`. This binary is tracked in the pinned upstream tree (`git ls-files pretrained_models/alexnet-bn.pth`), and upstream status remains clean. Raw audit finds exactly five missing `layer1..5.1.num_batches_tracked` scalar counters, no unexpected keys. Raw strict key match is **false** and remains reported truthfully.

Lead explicitly approved one minimal compatibility mapping: insert those **exact five** missing counters as scalar int64 zero, then strict-load the complete normalized mapping into a **separate disposable backbone**. Every original learned parameter, bias, running mean/variance and original tensor is retained bit-exact; file SHA remains unchanged; per-original-tensor hashes and inserted counter names are recorded. Missing learned/running state, unexpected keys, wrong shape/dtype, partial counter sets and cumulative-average BN are rejected in tests. Fixed BN momentum0.1 uses EMA independent of num_batches_tracked; `momentum=None` cumulative scheduling is unsupported and fails closed.

This is asset readiness validation only: no pretrained backbone tensor enters synthetic optimization. The released general_model tracker checkpoint is not loaded or fine-tuned. A strict backbone asset audit cannot establish pretraining leakage provenance or production training readiness.

## Canonical real-data binding and no-launch proposal

[Proposed config](<F:/Code/HiFT/configs/B2-wave3-proposed.json>) fixes source-derived settings and explicit deviations and remains launch_allowed=false. Binding imports the source-audit worker's canonical `validate_training_manifest(...,verify_files=True)`, requires authentic COCO2017/train2017 `hift-training-data-v1` frozen_preflight_only provenance, preserves candidatePneg0.05/no replacement/unknown-alias rejection, and pins the exact training manifest and reservations hashes. It rejects evaluation registries or synthetic substitutes. A frozen preflight-only manifest is **not** a validated real training distribution or legal/global-physical-identity certification.

The config now binds the authentic [frozen training manifest](<F:/Code/HiFT/artifacts/wave3-training/FROZEN-001/training_manifest.json>) SHA256 `5e1bd0151bcbdaa49fc40c4a7e21b11c1018c774cee793b7308d8e3d0749dabc`, reservations SHA256 `a4f0b91e0c94181f4f8dbc09a09762fa722d97cead728bd553c333447f508e7c`, and policy SHA256 `bb2789acac81f4d0401da68f0f1ba3b4d94dec19fbd91f1adea4a0b25df33309`. Bound proposed recipe SHA256 is `3caf3ed6009629b9e81da03903732f3cdecc7bb852aa6b773f9ae2f2ce2cfdf0`; unbound source-derived recipe SHA256 is `db5c267dc75f17f2bf9922fc6ee41899e57342396c834e212752de47e66fe1c8`. Config explicitly copies all18 canonical source hashes and validation checks them against both manifest and current files; any source/policy/reservations/config drift fails closed.

This is32 official COCO TRAIN images/209 canonical annotation instances, **zero temporal tracking videos**, not the released four-dataset mix. Source-audit's1024 candidate preflight accepted973 positives and rejected all51 requested negatives as unknown cross-image physical aliases; zero replacements and **zero accepted negatives**. Candidate negative fraction51/1024=0.049805 preserves the predeclared0.05 Bernoulli policy, but accepted distribution is not usable as the intended negative-training distribution. Canonical annotation-instance IDs do not establish global physical identity. Therefore config status remains PROVISIONAL_PREFLIGHT_MANIFEST_BOUND, validated_real_data=false, launch_allowed=false, G2/G5 PARTIAL. No data from this manifest enters the synthetic optimizer.

## Cost evidence reused, not reprofiled

The [Wave2 report](<F:/Code/HiFT/reports/wave2_training.md>) and [immutable CUDA-b profile](<F:/Code/HiFT/artifacts/profiles/wave2-cuda-20261023-b/manifest.json>) measured resident synthetic batch1/2/4 clipped-SGD throughput39.00/74.34/128.22 pairs/s and peak allocated176.10/191.22/307.22MiB on RTX5070Ti. Those measurements used a fixed late-phase proxy, not this new warmup/accumulation lifecycle. **No profiling/GPU rerun in Wave3 task9.**

The earlier graph-only1000-update effective16 estimate at microbatch4 was124.79s; effective220 proxy1.716s/update. A **hypothetical separately authorized short pilot, not run or currently launchable**, could bound100 effective220 updates (22,000 pairs; microbatch4×55). Applying the old resident-graph mean31.19666ms to5500 microbatches gives **171.58s /2.86min graph-only arithmetic**. This even charges W2's clip+SGD on every microbatch while actual accumulation would clip/step once per effective update; it is neither a wall-time upper nor lower bound. Physical source batch220 across two GPUs and physical microbatch4 with55-way accumulation are not BN/dropout/objective-equivalent.

The proposed full-window FP32 crop buffer alone is `220*3*(127^2+287^2)*4` bytes≈**248MiB host memory**, excluding targets, decoded images, workers and queues. Estimates exclude buffering, loader/decode/augmentation/H2D, validation/checkpoint I/O and actual global accumulation semantics. Real wall-time/budget needs those measurements and corrected data/recipe acceptance first. Checkpoint serialization and this correctness harness's few-second runtime are not training throughput. No70-epoch wall-time, convergence, or accuracy estimate is justified.

## Review / next decision

Lead reviewed global reductions, same-partition synthetic replay and proposed deviations, requested stricter optimizer state checks, and explicitly approved the five-counter-only backbone compatibility. Final acceptance still requires the revised evidence/config review. Recommendation for this worker: **continue data/recipe correctness**, not infer a B2 pilot grant from synthetic lifecycle or GPU capacity. Preserve separate G2/G5 partial statuses until actual data/recipe conditions are accepted.
