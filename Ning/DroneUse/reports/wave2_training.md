# Wave 2 — isolated corrected B2 training feasibility

**Measured, bounded synthetic feasibility only.** No full training, real-data throughput, convergence, accuracy, checkpoint reproduction, B0/B1 improvement, or maximum-batch claim. Task-6 scope only; upstream unchanged; no commit. Lead explicitly granted and later re-granted the exclusive GPU token, released immediately after each bounded process.

## Evidence and exact identity

Primary corrected evidence: [CUDA manifest](<F:/Code/HiFT/artifacts/profiles/wave2-cuda-20261023-b/manifest.json>) and [CPU manifest](<F:/Code/HiFT/artifacts/profiles/wave2-cpu-20261023-b/manifest.json>). These ignored artifact directories are created exclusively and never overwritten; each includes start/final manifests, per-batch raw timings/losses/gradient names, dirty diff, exact project/upstream source snapshots, package/build metadata, and output hashes. Failed batches would retain explicit `oom`/`error` status with no invented throughput.

- Assignment/runtime HEAD: `5ce4a432aaa3027c7a45d4c678ae04e27a20223a`; immutable upstream `992aa7c1605360d607af84cb804b93cd8ee81caf`.
- Primary CUDA run source-tree SHA256: `4ab78ac9b00bbf265f5cb9143f94de496e130c596a0d482cba90d7b88b592511`; tracked dirty-diff SHA256: `604966a14901dc489d2f4b59bc1fe18ffcc95b37a7c219d84ef072d102504580`. The source-tree hash also covers new/untracked adapters; a diff hash alone is insufficient. Concurrent workers' unrelated dirty files are recorded/snapshotted, not attributed to this task.
- [Released experiment config](<F:/Code/HiFT/third_party/HiFT/experiments/config.yaml>) SHA256 `41db70a6239e34780509c1da04c6daaa8f3efce014c362557514760efcefd956`; [profile plan](<F:/Code/HiFT/experiments/wave2_training_plan.json>) snapshot SHA256 `e3b17024a1954ee7391f8f3bb2440573ddd0a07a0a71b646c3d7ab57b94c0fa3`.
- [Official checkpoint](<F:/Code/HiFT/artifacts/checkpoints/general_model.pth>) exists, SHA256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`, **not loaded**. Model is explicitly randomly initialized. Layout correction changes learned computation even B=1; identical key/shape compatibility is not baseline accuracy compatibility.
- Python 3.13.0, torch 2.10.0+cu128, CUDA build12.8, NumPy2.4.4, Windows10 build19044; CPU Intel64 Family6 Model183 Stepping1; one CPU intra-op thread. NVIDIA RTX5070Ti, compute12.0, reported 17,094,475,776 device bytes, driver591.74. Global `nvidia-smi` start snapshot4065MiB used includes unrelated desktop/process residency; it is not this graph's allocator usage.
- Seed20261023; FP32; CUDA matmul and cuDNN TF32 disabled; deterministic algorithms/cuDNN enabled, benchmark autotune disabled. This controls this bounded run, not a promise of cross-platform bitwise equivalence. Primary CUDA timing began2026-10-23T11:06:10Z.

Historical [CUDA-a manifest](<F:/Code/HiFT/artifacts/profiles/wave2-cuda-20261023-a/manifest.json>) and [CPU-a manifest](<F:/Code/HiFT/artifacts/profiles/wave2-cpu-20261023-a/manifest.json>) remain immutable. They predate composition-level raw-logit/raw-regression mask ordering and must not be substituted for primary revised evidence. Their normal finite synthetic cases passed, but invalid inactive values had not been made safe before nonlinearities. Revised timing differences are not controlled speed comparisons.

After the primary profile, the target adapter acquired only additional invalid-input guards (integer/bounded configuration and float32-representable bbox range). The compact audit records its source hash as different, rather than claiming the final checkout is byte-identical. Lead reviewed the exact diff: normal target arithmetic is unchanged, and synthetic target construction occurs outside timed stages. Graph, profiler, layout, classification and masked-loss source hashes still match the measured snapshot. No GPU measurements were overwritten or attributed to the new guard code.

## Actual graph and disclosed deviations

[Graph](<F:/Code/HiFT/src/hift_research/training/graph.py>) uses actual pinned AlexNet and HiFT class bodies, not a toy backbone: 127/287 crops; three depthwise correlations;192-channel11x11 maps;121-token six-head transformer with one encoder/two decoder layers,2048-wide feedforward; localization/two classification heads. [Isolated loader](<F:/Code/HiFT/src/hift_research/training/upstream.py>) executes those class bodies and changes exactly two positional `arange(...).cuda()` calls to the embedding device. No upstream file is edited or patch applied.

Explicit B2 changes: bound [corrected encoder layout](<F:/Code/HiFT/src/hift_research/corrections/tensor_layout.py>); [corrected positive support / identity-checked pairs](<F:/Code/HiFT/src/hift_research/corrections/positive_targets.py>); [dense known-negative policy](<F:/Code/HiFT/src/hift_research/corrections/pair_targets.py>); [singleton-safe balanced NLL](<F:/Code/HiFT/src/hift_research/corrections/classification_loss.py>); [masked IoU/BCE](<F:/Code/HiFT/src/hift_research/corrections/masked_losses.py>). Source-audit owns these corrections; training imports rather than duplicates them. The ordinary bbox95,95,223,223 does not activate the tiny-support heuristic. Negative identities are explicitly distinct synthetic `(dataset,video,track)` keys, not a claim of a validated real sampler.

Dynamic regression decode uses actual batch/spatial sizes and tensor device, grid origin63/stride16, scale143, and released inverse-tanh xyxy mathematics. Only inputs outside(-1,1) saturate to±0.99, preserving released discontinuity; e.g.0.999 is not silently clipped. No in-place modification. `log1p` is mathematically equivalent to released `log(1+x)`, not bit-exact; CPU comparisons allow documented floating tolerance. Upstream IoU eps/+eps remain. Negative predicted extents are clamped to zero area by the B2 safe-loss adapter; positive-extent finite geometry retains source math.

Masks select raw regression before inverse-tanh and ignored cls1 raw logits before log-softmax. Inactive NaN/Inf cannot poison backward through `0*NaN`; invalid active raw values fail closed. BCE is dense upstream-style mean with explicit target/shape validation. Loss: `2.4*IoU + 0.5*positive-NLL + 0.5*negative-NLL + mean-BCEWithLogits`.

**Late-phase proxy, not the exact released training launcher:** layers1-2 weights and BN statistics frozen; layers3-5 weights/BN train; all grader parameters trainable. Model train mode retains transformer dropout0.1 and channel-attention dropout0.5. SGD backbone/grader learning rates0.0005/0.005, momentum0.9, weight decay0.0001, gradient clip norm10; no scheduler, early frozen-backbone phase, optimizer resume, AMP, accumulation, DataParallel, DDP, or multi-GPU execution. There are11,066,312 parameters,10,416,008 marked trainable. The source recipe uses70 epochs/global batch220/two GPUs; this profile does not execute that recipe. All runtime work is one process on one GPU, batches serialized.

Synthetic crops are resident uniform[0,255] FP32 tensors, matching unnormalized input scale but **not image statistics**. Positive and negative backward probes precede warmup. B1 alternates positive/negative minibatches; five timed repeats contain three positives/two negatives. B2/4 contain a50/50 mix, intentionally stress-testing masks rather than reproducing source NEG=0.05.

## CPU correctness gates

[Tests](<F:/Code/HiFT/tests/test_training_profile.py>) execute CPU B1/2/4 train-mode real forward/backward/SGD, representative head shapes, strict state-dict roundtrip, dynamic/rectangular decode and released-method comparison, saturation gradients/no mutation, malformed inputs, all-negative localization zero, raw regression NaN allzero-mask finite backward, ignored cls1 NaN finite backward, active-invalid rejection, profile token gate and output-reuse refusal.

Commands before GPU use:

```bash
.venv/Scripts/python.exe -m pytest tests/test_source_audit.py -q
# 36 passed, 3.67s before first profile
.venv/Scripts/python.exe -m pytest tests/test_training_profile.py tests/test_wave2_correctness.py -q
# 73 passed, 6.18s before revised profile (15 training +58 source-correction cases)
.venv/Scripts/python.exe -m pytest tests/test_training_profile.py tests/test_source_audit.py tests/test_wave2_correctness.py -q
# Final integrated CPU gate: 122 passed, 9.28s; upstream status remained clean
```

Each profiled batch has positive and negative probes with finite losses/gradients. Positive active parameter tensors have nonzero gradients; final positive/mixed iterations have no unexpected missing/zero/nonfinite parameter gradients. Expected exceptions are explicitly enumerated in raw JSON:

- Four `encoder.layers.0.eles` convolution/GroupNorm parameter tensors are genuinely dormant in the pinned forward and receive no gradients; retained, not silently removed.
- Four channel-attention projection parameter tensors have zero gradient at initialized gamma=0, but gamma itself is active. They become nonzero after the bounded SGD steps.
- Eight layers1-2 parameter tensors are deliberately frozen.
- Negative-only probes have zero localization loss and14 zero localization-head parameter gradients; these are expected policy consequences, not classifier inactivity. The generic raw `unexpected_zero` field lists them in negative probes; it is empty for positive probes/final positive or mixed iterations. Both classifier heads remain active. Nonzero tensor gradients do not imply every embedding row or every scalar is active.

## Measured latency, throughput, memory

[Profiler](<F:/Code/HiFT/src/hift_research/training/profiling.py>) / [CLI](<F:/Code/HiFT/scripts/profile_training.py>): two SGD warmup iterations and **five timed SGD repeats per batch**. Synchronize before/after each stage; wall latency includes launch, synchronization, validations inside losses, zero-grad, clipping, optimizer, and Python overhead. Per-parameter audit and scalar reporting are outside measured stages. Throughput is timed pairs divided by sum of total elapsed step times, not inverse median. Five repeats are a bounded feasibility sample with visible jitter, **not a stable performance benchmark**.

| Batch | Step mean / median / range (ms) | Synthetic pairs/s | Current allocated / reserved (MiB) | Peak allocated / reserved (MiB) | Status |
|---|---:|---:|---:|---:|---|
|1|25.64 /25.06 /23.78–29.95|39.00|142.37 /238|176.10 /238|measured, no OOM|
|2|26.90 /26.53 /25.88–28.07|74.34|142.89 /270|191.22 /270|measured, no OOM|
|4|31.20 /30.13 /30.10–34.09|128.22|145.54 /388|307.22 /388|measured, no OOM|

Allocator peaks reset after warmup, so include resident tensors, parameters, momentum buffers, gradients and timed transient allocations, but not CUDA context/driver/global residency. Reserved cache may carry probe/warmup history. Allocated/reserved are not process VRAM; CPU memory is explicitly N/A, not fabricated. OOM was not encountered; **maximum fitting batch and long-run memory were not tested**, and tiny observed peaks must not be extrapolated to global batch220.

Median stage milliseconds:

| Stage | B1 | B2 | B4 |
|---|---:|---:|---:|
|Template backbone|0.92|0.93|1.21|
|Search backbone|1.25|1.79|3.31|
|Correlation + transformer + heads|5.70|5.33|5.55|
|Decode + losses + fail-closed validation|4.94|5.21|5.48|
|Backward|9.48|10.57|11.62|
|Gradient clipping|1.34|1.29|1.35|
|SGD|1.39|1.25|1.36|
|Zero-grad|0.15|0.16|0.16|

Separate medians do not necessarily sum to total median. Actual attention and all heads are included; transformer-only cost, correction-only cost, and overhead versus B0 were **not isolated**. No attention/layout speedup claim follows. Mask validation has host/device synchronization cost; optimizing it is deferred until correctness integration.

Revised single-thread CPU synthetic throughput: B1 **2.75**, B2 **3.10**, B4 **2.28** pairs/s. CPU kernel behavior/jitter differs from GPU; neither these numbers nor the GPU numbers measure a full input pipeline or constitute cross-device training recommendations.

## Planned microbatch/accumulation/pilot estimates — not measured training

Prefer **microbatch4 as a bounded pilot starting candidate**, because it succeeded and had highest measured synthetic throughput among tested sizes, not because it is maximum capacity. Accumulation has **not** been implemented or timed. For effective batchE, candidate accumulation isE/microbatch when divisible; divide correctly normalized losses and clip/step once per effective update. Accumulation changes BN, dropout, and class-balanced/weighted-loss reductions unless explicitly controlled: simply averaging mixed microbatch losses is not guaranteed equivalent to global loss normalization, especially varying positive masks. It is not a reproduction of the source batch220.

Linear graph-only proxy using measured mean step latency, charging an SGD+clip on every microbatch (real accumulation would change this), ignoring pipeline/validation/checkpoint costs:

| Microbatch | Accumulations for E=16 | Planned1000 updates at E=16 | Accumulations for E=220 | Graph proxy seconds/E=220 update |
|---|---:|---:|---:|---:|
|1|16|410.21s /6.84min|220|5.640|
|2|8|215.21s /3.59min|110|2.959|
|4|4|124.79s /2.08min|55|1.716|

These are arithmetic estimates, neither budget authorization nor defensible wall-time upper/lower bounds. The1000-update effective16 pilot would process16,000 synthetic-pair equivalents; no pilot/full training was run. True cost requires real data I/O/decode/augmentation/loader/H2D, sampler distribution, training schedule, convergence diagnostics, validation, checkpoint writes, and measured accumulation semantics. No70-epoch estimate is justified without a frozen training-split size and those measurements.

## Acceptance and remaining work

**Delivered:** isolated actual B2 graph; dynamic/device-safe decode; CPU mathematical/composition tests; bounded sequential B1/2/4 synthetic forward/backward/clipped-SGD measurements; honest active/dormant gradients; allocated/reserved memory; exact source/config/checkpoint/environment identities and immutable evidence; bounded pilot estimates. Integration-owner review requested; lead identified and approved mask-order fixes before the revised measurement.

**Not delivered / separate gates:** independently reviewed complete B2 optimizer/scheduler and augmentation recipe; dataset acquisition/licensing and sequence-disjoint split/leakage checks; real identity-aware negative sampling; long-run mask/target frequencies and convergence; strict checkpoint-initialized training experiment; validated accumulation/global-reduction/BN policy; full pipeline timing; controlled retraining and B2 accuracy; R1 training. No official checkpoint mutation, upstream edits, or commits.
