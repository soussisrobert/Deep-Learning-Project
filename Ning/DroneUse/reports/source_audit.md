# Pinned HiFT source audit

**Status:** bounded source audit; synthetic CPU correctness evidence only. No training, dataset evaluation, accuracy/speed claim, or checkpoint reproduction. Upstream SHA independently verified as `992aa7c1605360d607af84cb804b93cd8ee81caf`; upstream was not edited.

## Findings and baseline separation

| ID | Classification | Evidence / consequence | Repair status and checkpoint impact |
|---|---|---|---|
| SA-01 | Proven axis/index bug | Encoder reinterprets `[S,B,C]` as `[B,C,H,W]`; changes B=1 spatial/channel mapping and leaks samples at B=2/4 | Isolated adapter + unapplied patch; B2, not a B1 official-checkpoint-compatible correction |
| SA-02 | Proven supervision disconnect | Negative pair sampling never reaches target generator; both classifiers and localization receive positive-object supervision | Explicit dense-background B2 policy + patch; changed training distribution, requires controlled retraining |
| SA-03 | Proven unsupported launch | Public initialization unconditionally returns rank 0 / world size 1 | Fail-closed adapter + patch; **not** a distributed-training implementation |
| SA-04 | Proven singleton criterion bug | `.squeeze()` turns exactly one selected index into a scalar; criterion discards it | Isolated criterion + patch; changes training edge cases, no parameter/key changes |
| SA-05 | Proven target-generator edge failure | A positive 1×1-pixel box can have empty radial support and raises a reduction error | Reproduction test only; frequency and appropriate positive-target policy unresolved |

B0 must preserve released computation. B1 inference-only corrections must remain independently separable. SA-01 changes computation even with one sample and zero-initialized channel-attention gain: unchanged checkpoint key/shape compatibility is **not** semantic compatibility. SA-01/02/04 belong to a new B2 controlled-training recipe, not a relabelled released checkpoint.

## 1. Transformer: complete shapes and index mapping

### Source trace

Default crop sizes 127 and 287 are declared in [config](<F:/Code/HiFT/third_party/HiFT/pysot/core/config.py#L46-L52>); released experiment uses output 11 in [experiment](<F:/Code/HiFT/third_party/HiFT/experiments/config.yaml#L25-L31>). The actual [AlexNet implementation](<F:/Code/HiFT/third_party/HiFT/pysot/models/backbone/newalexnet.py#L152-L198>) uses conv11/stride2, pool3/stride2, conv5, pool3/stride2, then three unpadded conv3 layers. Static convolution arithmetic gives:

| Stage | Template shape | Search shape |
|---|---|---|
| Input | `[B,3,127,127]` | `[B,3,287,287]` |
| layer1 | `[B,96,29,29]` | `[B,96,69,69]` |
| layer2 | `[B,256,12,12]` | `[B,256,32,32]` |
| layer3 / returned first | `[B,384,10,10]` | `[B,384,30,30]` |
| layer4 / returned second | `[B,384,8,8]` | `[B,384,28,28]` |
| layer5 / returned third | `[B,256,6,6]` | `[B,256,26,26]` |
| Depthwise correlation | — | three maps `[B,384/384/256,21,21]` |
| Each fusion projection | — | three maps `[B,192,11,11]` |
| Positional embedding | — | `[B,192,11,11]` (96 column + 96 row channels) |
| Three transformer inputs | — | `[121,B,192]` each |
| One encoder / two decoder layers | — | `[121,B,192]` throughout |
| Restored decoder feature | — | `[B,192,11,11]` |
| Localization / cls1 / cls2 heads | — | `[B,4,11,11]` / `[B,2,11,11]` / `[B,1,11,11]` |

Correlation's grouped convolution uses `groups=B*C` and restores the same sample axis in [fusion lines 76–92](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L76-L92>). Projection kernels/channels are [lines 14–28](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L14-L28>). Position construction and outer map/sequence conversions are [lines 95–112](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L95-L112>): those outer conversions are correct. Heads are [lines 33–66](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L33-L66>) and [114–120](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L114-L120>). The transformer is configured as channels=192, heads=6, encoder layers=1, decoder layers=2 at [line 63](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/utile.py#L63>).

### Exact defect, including gamma=0

The sequence-first convention is supported by `MultiheadAttention` instantiation at [encoder line 214](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/tran.py#L212-L216>), decoder attention [lines 319–329](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/tran.py#L319-L329>), and transformer dimension checks [lines 67–76](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/tran.py#L67-L76>).

After attention and layer normalization, [encoder lines 244–255](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/tran.py#L244-L255>) compute dimensions from a **temporary permuted view**, but do not retain that permutation. Lines 250–251 then call `src.view(B,C,H,W)` and `srcc.contiguous().view(B,C,H,W)` on logical `[S,B,C]`. A reshape cannot exchange semantic axes.

For contiguous sequence `A[S,B,C]`, write `q=(b*C+c)*S+s`. The upstream gamma=0 channel-attention branch plus its return conversion produces

```
O[s,b,c] = A.ravel()[q]
        = A[q // (B*C), (q // C) % B, q % C]
```

instead of `A[s,b,c]`. Channel attention is initialized with `gamma=0` and returns its first map unchanged at [lines 24–32](<F:/Code/HiFT/third_party/HiFT/pysot/models/utile/tran.py#L24-L32>), but the entry/exit conversions do **not** undo each other. At B=1 this reindexes spatial/channel content. At B=2/4, any nominal output sample contains source elements from every actual sample. A nonzero channel-attention gain additionally pools/weights incorrectly grouped data; it is not needed to expose the bug.

Correct conversion: `A.permute(1,2,0).contiguous().view(B,C,H,W)`, then flatten spatial axes and permute back. Both source streams need the correction. The [isolated layout adapter](<F:/Code/HiFT/src/hift_research/corrections/tensor_layout.py>) changes no parameters and explicitly rejects non-square sequence length in the pinned encoder; standalone conversion helpers also support rectangular maps. [Patch 0001](<F:/Code/HiFT/patches/0001-transformer-layout-b2.patch>) is unapplied.

### Test scope

[Audit tests](<F:/Code/HiFT/tests/test_source_audit.py>) cover exact full-size `S=121,C=192` NumPy index mapping and outer roundtrips at B=1,2,4, sample-ID tracing at B=2,4, and other-sample perturbation. Executed torch tests instantiate the **actual pinned** Transformer(192,6,1,2), in CPU eval mode, and verify shapes, finite values, differing original/corrected outputs at B=1/2/4, unchanged state dictionaries, full-transformer sample isolation at B=2/4, and repaired alone-vs-batched behavior. At both initialized gamma=0 and explicit nonzero gamma=0.3, changing another sample changes upstream sample-zero outputs by more than 1e-5; corrected sample-zero outputs are exactly unchanged (rtol=atol=0). The corrected batched result agrees with its B=1 execution within rtol=1e-5 / atol=2e-6.

Separate executed CPU tests run the actual AlexNet and HiFT class bodies at B=1/2/4, confirming returned backbone, correlation, and all three head shapes in the table. The sole compatibility shim substitutes CPU indices for the two positional `arange(...).cuda()` expressions; no model mathematics or parameters are rewritten. Eval mode deliberately excludes dropout and normal training BatchNorm coupling; no claim of independent samples through a train-mode backbone is made.

## 2. Negative pairs: sampling → labels → loss/masks

1. [Dataset lines 233–241](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/dataset.py#L233-L241>) sample `neg` then obtain two random targets when true. The configured probability is 0.05 in [experiment line 81](<F:/Code/HiFT/third_party/HiFT/experiments/config.yaml#L81>).
2. Augmentation returns the search object's bbox; [dataset lines 261–267](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/dataset.py#L261-L267>) call `anchor_target.get(bbox,size)` **without** the negative flag. [AnchorTarget line 30](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L30>) accepts no negative argument.
3. [Target lines 41–66](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L41-L66>) give the unrelated search object positive localization weights. [Lines 78–86](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L78-L86>) give it a soft positive cls2 map. [Lines 103–108](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L103-L108>) give it hard cls1 positives plus sampled background; -1 and -2 are ignored by the cls1 criterion.
4. [Model lines 95–119](<F:/Code/HiFT/third_party/HiFT/pysot/models/model_builder.py#L95-L119>) consume those fields unchanged. `bbox` itself is loaded but does not provide an alternate negative mask. cls1 selects label==1/0 in [loss lines 23–30](<F:/Code/HiFT/third_party/HiFT/pysot/models/loss.py#L23-L30>); cls2 uses ordinary unmasked `BCEWithLogitsLoss` created at [model line 24](<F:/Code/HiFT/third_party/HiFT/pysot/models/model_builder.py#L22-L25>). Localization receives `weightxff`; the IoU criterion [lines 59–65](<F:/Code/HiFT/third_party/HiFT/pysot/models/loss.py#L59-L65>) suppresses regression only if that weight is zero. No sampled-negative flag or alternate mask exists downstream.

The deterministic test executes the actual `TrkDataset.__getitem__` and `AnchorTarget` AST nodes unchanged, replacing only image I/O/augmentation and metadata. With NEG=1 and deliberately distinct fake template/search identities, it proves cls1 positives, cls2 peak=1, and nonzero regression weight. With zero logits, BCE derivative `sigmoid(0)-target` is negative at the unrelated object's positive peak: the supervision attracts rather than rejects it. This proves the disconnect without requiring any real dataset.

[Pair-target adapter](<F:/Code/HiFT/src/hift_research/corrections/pair_targets.py>) and [patch 0002](<F:/Code/HiFT/patches/0002-negative-pair-supervision-b2.patch>) define one explicit policy: for a known negative, cls1=0 everywhere, cls2=0 everywhere, localization labels=0 placeholders, regression weights=0. Positive pairs delegate unchanged. The upstream cls1 criterion retains its 0.5 negative-class weighting, and cls2 remains a mean over all cells. Dense negative labels rather than NEG_NUM spatial subsampling is a disclosed research choice. This is not proof of the authors' intended correction or of an accuracy benefit.

**Sampling caveat / hypothesis:** random-target sampling does not guarantee distinct object identities. Its collision frequency depends on dataset metadata and sampler composition and was not measured. A future identity-aware sampler should reject false negatives; doing so is a separate distribution change, not silently included here.

## 3. Classification and target-generator edge cases

### SA-04: singleton selected class silently loses its term

[Loss lines 26–27](<F:/Code/HiFT/third_party/HiFT/pysot/models/loss.py#L26-L27>) apply `.nonzero(...).squeeze()`. Exactly one selected item becomes a zero-dimensional index tensor. [Loss lines 13–20](<F:/Code/HiFT/third_party/HiFT/pysot/models/loss.py#L13-L20>) return integer zero whenever the selection is scalar, dropping a valid observation. The torch test invokes the actual helper with exactly one positive index and checks its result against an isolated balanced NLL correction. [Criterion adapter](<F:/Code/HiFT/src/hift_research/corrections/classification_loss.py>) preserves class balance, ignores -1/-2, and returns differentiable zero for an empty class; [patch 0004](<F:/Code/HiFT/patches/0004-singleton-class-selector-b2.patch>) minimally flattens selectors and follows the prediction device. Actual frequencies in batches are not measured.

### SA-05: tiny positive boxes can crash target construction

For bbox `(143,143,144,144)` and 11×11 output, clamped integer grid endpoints coincide. Then `w=h=1` and the radial cutoff is zero. Strict `<0` selects no elements; [target lines 47–59](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L47-L59>) call `se.min()`/`se.max()` on the empty array. An actual-source NumPy test asserts this ValueError. The analogous cls2 reduction is at [lines 81–83](<F:/Code/HiFT/third_party/HiFT/pysot/datasets/anchortarget.py#L81-L83>). Whether real augmentation reaches these cases, and whether the correct policy is resampling or minimum support, remain unresolved. No speculative positive-box repair is included.

## 4. Distributed initialization

The active public [dist_init lines 102–115](<F:/Code/HiFT/third_party/HiFT/pysot/utils/distributed.py#L102-L115>) comments out `_dist_init` and unconditionally returns `(0,1)`. A CPU AST execution with WORLD_SIZE=4/RANK=3 returns `(0,1)` and sets `inited=True`, proving ignored launcher metadata.

Consequences follow directly from callers:

- [Training loader lines 64–71](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L64-L71>) never enters DistributedSampler under the active path.
- [Reduction lines 130–141](<F:/Code/HiFT/third_party/HiFT/pysot/utils/distributed.py#L130-L141>) never synchronize processes because world size is reported as one.
- The trainer uses [DataParallel lines 228–230](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L228-L230>) and [264–270](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L264-L270>), not DDP. Each separately launched process would believe it is the writer rank and can contend for the same [snapshot paths](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L122-L137>). This is a predicted source consequence; no multi-process/GPU training was launched.
- The dormant [_dist_init lines 83–89](<F:/Code/HiFT/third_party/HiFT/pysot/utils/distributed.py#L83-L89>) hardcodes rank zero, selects `rank % num_gpus`, and requests NCCL. Simply uncommenting it is not a complete repair: launcher rank/local-rank handling is wrong, and CPU with zero visible GPUs would divide by zero. `--local_rank` is parsed at [trainer lines 43–44](<F:/Code/HiFT/third_party/HiFT/tools/train.py#L43-L44>) but not used for device selection.

[Guard adapter](<F:/Code/HiFT/src/hift_research/corrections/distributed_guard.py>) and [patch 0003](<F:/Code/HiFT/patches/0003-reject-unsupported-distributed-launch.patch>) reject WORLD_SIZE≠1, RANK≠0, or LOCAL_RANK≠0. They preserve documented one-process DataParallel operation; they do not initialize process groups, replace DataParallel, validate Slurm metadata, or claim multi-node support.

## 5. Verification record and limits

Initial CPU NumPy environment: Python with NumPy 2.4.4 / pytest 8.4.2; torch absent. Command:

```bash
PYTHONPATH=src python -m pytest tests/test_source_audit.py -q
```

**Measured initial result:** 23 passed, 8 optional torch tests skipped, 0.29 s. No training or upstream imports were required for those 23 tests: selected AST execution preserves the audited function/class bodies while stubbing only dependencies explicitly described above.

Final isolated environment: Python 3.13.0 (Windows AMD64), NumPy 2.4.4, pytest 8.4.2, torch 2.10.0+cu128. **All audit tensors and operations run on CPU; no GPU or training is used.** Seed=20261023; actual torch models are randomly initialized and in eval mode, not loaded from a checkpoint. Thread count=1 for deterministic bounded model tests.

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_source_audit.py -q
```

**Measured final result: 36 passed, 0 skipped, 4.13 s.** This includes the original 23 NumPy tests plus 13 actual-source torch tests: full AlexNet/correlation/HiFT shapes (3), full Transformer original-versus-corrected outputs/state dictionaries (3), full Transformer isolation at two batch sizes and two gamma values (4), actual IoU and BCE mask gradients (1), singleton NLL reproduction/correction (1), and layout helper roundtrip/invalid-input checks (1). Test duration is a verification log, not an inference-speed measurement.

All four unapplied patches pass `git -C third_party/HiFT apply --check ...`; `git -C third_party/HiFT status --porcelain` is empty after patch generation/checking and again after the final tests. See [patch instructions](<F:/Code/HiFT/patches/README.md>) and [generator](<F:/Code/HiFT/patches/build_source_patches.py>). No source patch was applied to upstream.

**Deliberate boundaries:** no upstream edits, no root configuration changes, no evaluation/literature work, no official-checkpoint mutation, no training. The shape table is derived statically and its backbone/correlation/head endpoints are independently checked with actual-source CPU execution. Measured torch tests use random synthetic inputs and evaluate only correctness invariants. This report does not audit optimizer schedules, all augmentation edge cases, complete DDP migration, real negative-collision rates, or checkpoint training provenance.
