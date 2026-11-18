# HiFT Wave4 — authentic temporal supervision and bounded pilot readiness

**Verified delivery:** all439 integrated tests passed (zero failures/errors/skips), dependencies are consistent, all33Wave4 reference hashes and29critical executing source files verify. All140protected prior working files remain byte-identical to the Wave4 start, all26prior referenced artifacts match, and upstream is clean. The final bookkeeping commit and its exact clean ending HEAD are reported in the handoff receipt/response.

**Decision:** recommend **GO only for a separately human-authorized, grader-only corrected-B2 reliability pilot**: at most100 effective updates/1600 indexed attempt slots/900seconds. This wave performed **zero real-image backward calls or optimizer steps**, started no pilot and implemented no R1. GO is not permission to train, a reproduction claim, a learned B2 checkpoint, or a tracking-accuracy result.

## A. Mandate, preserved baseline and initialization

- Work continued from clean Wave3 delivery `28e9e197c62a21450587465e4c32abdf97fb88d3`; the repository was not reinitialized. Immutable upstream remains `992aa7c1605360d607af84cb804b93cd8ee81caf` and clean.
- [Preservation verifier](<../scripts/verify_wave4_preservation.py>) compares **all140 protected prior tracked files** to their historical Git blobs and actual starting working-byte SHA256s, all21Wave3+5Wave2 referenced artifacts, and the official tracker checkpoint. Git text filters explain the preserved initial raw-CRLF comparison false alarm; canonical equality is not substituted for the independent starting-byte check.
- Existing15sequence/2937frame B0/B1 evaluation, all configurations, predictions and adverse findings remain frozen. No benchmark, oracle, compatibility experiment or completed tracking run was repeated; no B1 promotion or retuning. The full historical benchmark and M1/M2 acceptance gaps do not disappear because this bounded pilot is ready.
- Official tracker checkpoint SHA256 `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101` was preserved, **not used for optimization**. Forward-only checks used strict source ImageNet backbone SHA256 `933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8` plus a newly random grader. All30 supplied tensors stay bit-exact; only five missing scalar int64 BN counters become zero. There is no silent partial load.

## B. Authentic source, identity and split

[Source investigation and acquisition report](<wave4_temporal_data.md>) records actual official VID/YTBB/MOT probes, identity semantics, license conditions and bounded-download receipts. VID pages and YTBB metadata are **available**; this is not an assertion that those corpora are universally unavailable. A ready, bounded, exhaustively annotated simultaneous-instance media subset was not established from those sources. YTBB segment keys are not global physical identity and different segments cannot automatically become negatives.

MOT17 official TRAIN was selected before pair outcomes: **TRAIN02/09/10; DEV05**,40original frames each, two20frame windows per video. All160original JPEGs decode in independent implementations. Complete original GT has **65,877 rows**;4,289 selected-frame annotation rows include1,645 mark-zero rows retained rather than erased. Accounted acquisition is **25.97MiB/128MiB**, not a full5.86GBarchive download.

The [MOT protocol](https://arxiv.org/html/2010.07548v2) uses author trajectory fragments: reentry/ambiguous occlusion can create a new ID. Identity is `(MOT17 release, canonical video, author fragment)`, **not a global person**. Same-ID endpoints are observed native frames, not interpolation; native1based x/y become x−1/y−1, width/height unchanged. Endpoints require valid person class/presence, visibility≥.5, nonzero image support, no duplicate/reuse ambiguity and **gap1..19**. Detector variants share a video; MOT16 and suspected MOT15 aliases are reserved, not counted as independent clips or numeric-ID-equated across releases.

DEV holds out the known ETH/Ess recording family. TRAIN09/10 are conservatively one ADL collection, not independent collection replications. Full SHA/pHash screens found zero known TRAIN/DEV or frozen15evaluation collisions. These checks do **not** prove universal re-edit, physical-person or pretraining independence. License is CC BY-NC-SA3.0 plus possible original-source conditions; this is bounded noncommercial research preparation, not blanket legal certification.

## C. Production pairs, real preflight and independent audit

[Pairing report](<wave4_pairing.md>) and [frozen policies](<../artifacts/wave4-preflight/FROZEN-002/freeze_receipt.json>) bind source, code/transitives, pools, reservations, content screen, seeds and eligibility before candidate outcomes. TRAIN requests negatives by indexed Bernoulli `.05`, seed20261023; no refill. Positives draw observed ordered endpoints of one verified fragment. Unknown cross-image/video aliases are never promoted to known-distinct identities.

A negative uses **two simultaneous distinct author tracks A/B in one frame**, valid person annotations and disjoint boxes. Merely differing IDs at different times is insufficient. TemplateA must be absent from the **actual final augmented search crop centered on B**, including JPEG/interpolation support:18pixels in511pre-JPEG coordinates, exact510-based inverse affine and2original-pixel margin. Postaugmentation intersection rejects the candidate; no fallback relabeling or favorable resampling. Dense negative classification is background and localization targets/weights are zero—an explicit repair, not the source's misleading positive negative-target behavior.

| Real cohort | Attempted | Accepted temporal positive | Accepted safe negative | Rejected | Replacements |
|---|---:|---:|---:|---:|---:|
| TRAIN preflight |1024|973|50|1|0|
| Fixed DEV control |128|64|64|0|0|

The single TRAIN rejection is **templateA visible in the actual augmented negative search**. Requested51/1024=4.9805% negatives, Wilson95%3.8082–6.4892%; accepted50/1023=4.8876%, descriptive Wilson3.7268–6.3859%. Requested prior, conditional eligible population and accepted prevalence are different quantities; these intervals are not generalization uncertainty over independent videos.

- Independent fresh production replay validated **all1152 records and1151 accepted array archives**, not a convenient subset. Lead's separate [original-GT/affine/footprint/grid audit](<../artifacts/wave4-integration/PAIR-ALGEBRA-001.json>) rederived endpoint annotations, simultaneous distinction, negative exclusion, class law, mask domains and all accepted offset targets without invoking the sampler's geometry routines. Maximum composed-affine error4.55e−13; encoded offset versus independent63origin/16stride/tanh equation error≤1.57e−7.
- TRAIN actual representation:120frames,53fragments,3videos. Positive gap median6/P9515/max19; native motion median10.77/P9570.33/max125pixels. [Full distributions and examples](<../artifacts/wave4-preflight/VERIFICATION-002/distributions_supplement.json>) include scale/aspect, native/resized size, crop support, augmentation and representation. Stationary/large-motion/rapid-scale/tiny/clipped/missing/reused-ID numerical fixtures are explicitly synthetic when those extremes are absent in the bounded authentic cohort; no fake real example is invented. Missing endpoints fail closed; unused missing intermediate annotations are recorded, never filled.
- All973positive targets have nonempty support. The retained source radial-weight boundary produces **298weight64cells/23363activecells=1.2755%, across98pairs**; this finite but unusual source heuristic is audited, not silently clipped or retuned. Historical511/510 target-label versus actual-pixel convention is disclosed; negative physical exclusion uses actual510pixel geometry.
- DEV is a **fixed128attempt weighted monitoring control**, alternating64positive/64negative with augmentation disabled, not128independent examples or a class-prevalence estimate. Its64negative draws contain **38unique source pairs/arrays,26repeats,max multiplicity4**, drawn without postoutcome balancing from50eligible pairs. DEV loss is paired supervision loss, never tracking AUC/precision/success.

Failed FROZEN/PREFLIGHT001 and REAL-LOADER001 are preserved. A NumPy boolean was not canonical JSON;002 converts only scalar metadata to native Python values. All20failed-source bytes were reconstructed and matched their preregistered hashes; first three positive and first negative arrays/numeric geometry are identical. There is no `default=str` masking workaround or changed seed/pool.

## D. Actual execution, bounded CUDA and safe recovery

[Lifecycle report](<wave4_lifecycle.md>) separates execution evidence from training outcomes.

1. Actual temporal decoding/crop/augmentation/target/collation was exercised under Windows serial and **spawn2workers**, differing worker-base seeds, ordered indices, consumed-cursor recovery and missing/decode/unexpected exceptions. [REAL-LOADER002](<../artifacts/wave4-integration/REAL-LOADER-002/manifest.json>) passes all6comparisons; failures stop with the original index, no hidden replacement, and clean reconstruction reproduces the batch. This is not a claim of exact asynchronous-prefetch resume.
2. Direct authentic NumPy transposes exposed a real defect missed by resident synthetic tensors and C-contiguous NPZ reloads: channels-last-strided Torch tensors reached the immutable grader's grouped-correlation `.view` and failed. [REAL-FORWARD001 failure](<../artifacts/wave4-feasibility/REAL-FORWARD-001/failure.json>) is retained. The new isolated [shared CPU NCHW bridge](<../src/hift_research/training/wave4_layout.py>) changes **storage only**, preserves every pixel bit, and rejects malformed dtype/shape/device/nonfinite input. No upstream/Wave3/frozen-temporal-file modification was made. The future controller and real profiler use the same bridge.
3. Exactly **16random-only CUDA effective SGD updates total**, serialized on one GPU:8device-lifecycle updates at source epochs0/10;4first shared-controller replay updates;4**new noncontiguous-production-input regression** updates after the observed failure. The explicit [pre-execution addendum](<../experiments/wave4_layout_and_dev_addendum.json>) raised the agent-internal12cap to16 before the new regression; this was not repetition of a successful tracking/benchmark experiment. No authentic array or pretrained tensor entered optimization.
4. Latest [production-layout CUDA proof](<../artifacts/wave4-lifecycle/production-layout-cuda-001/verification.json>) passes all13comparisons: model/BN buffers, momentum, clipped gradients, scheduler/LR, Python/NumPy/TorchCPU/CUDA RNG, attempt/loss trace and cursor are bit-exact. A midwindow crash performs no extra SGD commit, taints the instance and forbids checkpoint/continuation; reconstruct from an atomic effective-update boundary and replay that entire indexed window. Checkpoint save/load measured.452/.515seconds. Restricted safe tensor loading and hash/config/data/source/device checks reject mismatches.
5. New CPU tail test uses16then2accepted samples, actual global denominators1936then242BCEcells, exactly two zero-grad/clip/SGD calls and independent tail loss/clipped-gradient equivalence. All-rejected windows commit attempted indices and skip optimizer updates, never refill. Fewer100updates within1600slots is completionFAIL, not permission to extend training.
6. [REAL-FORWARD002](<../artifacts/wave4-feasibility/REAL-FORWARD-002/manifest.json>) passes actual first32TRAINattempts (**28positive/4negative**, not selected to force5%) and the **complete128DEV** path. Initial untrained DEV loss3.453117579 is finite; it is not accuracy, improvement or convergence evidence. Every model parameter is unchanged and all parameter gradients remain absent. Train-mode grader BN updates are disposable; no adapted checkpoint is saved. Failed4+successful160=164real forward attempts, within the preregistered208cap; **zero real backward/SGD**.

The recommended production mode is **num_workers0/no prefetch**, physical4attempt slots and a16attempt effective window. Global observed class/mask denominators normalize accumulated losses; clipping/step occur once per effective window. Accepted size can be below16. Microbatch BN/dropout is **not** a physical16or220batch equivalence. Existing source-stage CPU evidence and bounded CUDA epoch0/10 checks do not establish complete70epoch CUDA stage-transition/async-worker training; the proposed pilot deliberately remains epoch0.

## E. Frozen reference versus pilot, monitoring and measured resources

| Property | [B2-Reference](<../configs/B2-Reference-W4.json>) | [B2-Pilot](<../configs/B2-Pilot-W4.json>) |
|---|---|---|
| Purpose | Closest documented original70epoch recipe, not launchready | Future bounded grader-only reliability check |
| Data | Original four-source recipe remains unvalidated | Frozen MOT17TRAIN02/09/10, separateDEV05 |
| Clock |70epochs×600000attempts; original220batch/2GPUs | Sourceepoch0 unchanged;≤100updates/1600attemptslots |
| Optimizer | SourceSGD momentum.9,WD.0001,clip10; LR warmup/decay documented | GraderLR.0025,momentum.9,WD.0001,clip10; no compressed70epoch schedule |
| Trainable scope | Sourcefreeze/unfreeze stages documented | **Entire corrected grader**: fusion/transformer/prediction heads; backbone and backboneBN frozen |
| Batch | Source semantics, not reproduced here | Physical≤4accepted/4attemptslots; nominal≤16accepted/16attemptslots; accumulation≤4 |
| Authorization | Disabled | Disabled until a separate exact human grant+acknowledgment |

Seed20261023 versus source123456, singleGPU/microbatch BN, conditional same-frame safe negatives, tiny temporal source/cohort, source-valued head-only clock and stable optimizer groups are explicit pilot deviations. Stable groups retain grader momentum at stage transitions unlike the source optimizer rebuild; the pilot does not reach that transition. Reference/full reproduction is **not** declared ready.

DEV is evaluated at updates0/25/50/100. **Fixed-last100 is primary**; best-DEV is secondary with earliest-update tie handling and cannot rescue primary failure. Operational nondivergence/completion is distinct from the preregistered preliminary signal **fixed-last DEV loss≤.95×initial**. Otherwise no promising signal is claimed. Even a loss decrease cannot establish tracking gains, small-object/UAV generalization, or causal attribution to a particular correction; BN adaptation and the reduced recipe remain confounded. Nonfinite/OOM/identity drift/input faults/budget exhaustion stop; two consecutive DEV losses>3×initial stop. Do not retune against the frozen15sequence panel.

[Measured full-input budget](<../artifacts/wave4-feasibility/BUDGET-001.json>):

- Actual32candidate input mean **75.568ms/attempt**, CUDA forward/loss mean **40.770ms/physical batch**; input includes read/hash/decode, source mean-padded crop/JPEG, augmentation, geometry/targets/validation. Both decoded and prepared-crop LRUs are genuinely bounded at256MiB each; timing uses that production policy, not an unbounded resident cache.
- Complete fixed128DEV input+eval forward measured **5.891seconds**, included four times. Training backward/audit/clip/SGD cost comes from the new random-only shared-controller proof, not a fabricated real SGD measurement. Warm autograd-forward time floors the no-grad real-forward estimate; cold initialization and four checkpoint saves are included.
- Central projection **192.02seconds/3.20minutes**, engineering range **144.01–573.14seconds/2.40–9.55minutes**, **hard900seconds/15minutes**. This broad range is not a statistical confidence interval, a throughput guarantee, or a convergence forecast; the synthetic warm sample is only one effective window.
- On the16GBRTX5070Ti, actual real forward peak Torch allocation **138.718MiB**, reserved194MiB; actual process peak working set about2.02GiB. Synthetic training allocator measurements are recorded separately and include comparison-model overhead. These are not total driver/context VRAM or a long-run leak proof. Reserve the same machine/GPU exclusively and keep the configured memory stop; no AMP/offload/distributed equivalence is asserted.
- Manual recovery must account for **all previous failed-window, checkpoint and invocation wall time** against900seconds; a saved committed boundary does not itself include discarded failed work. No automatic retry or asynchronous resume is authorized. Review failure logs/remaining resources and obtain renewed explicit authorization before recovery.

## F. W4 acceptance gates and decision scope

| Gate | Outcome | Evidence and boundary |
|---|---|---|
| W4-G1 genuine temporal source/positives | PASS bounded |160original frames, official completeGT,973observed same-fragment nonzero-gap positives; no global-person claim |
| W4-G2 legitimate negatives | PASS bounded |Simultaneous A/B witness and actual guarded crop exclusion;50accepted/1rejected, unknown cross-video aliases excluded |
| W4-G3 auditable real preflight/independence | PASS bounded |1024TRAIN+128DEV all records/archives fresh-replayed; independent original-GT/grid audit; known15eval overlap0; conditional/repeated DEV limitations retained |
| W4-G4 production execution/recovery | PASS bounded serial fallback |ActualWindows loader/exception recovery; observed storage defect repaired; same-controller CUDA random-only boundary replay; actual full DEV+TRAIN forward only |
| W4-G5 frozen feasible pilot | PASS for recommendation only |Separate reference/pilot, hashes/cohort/900second budget, fixed-last monitoring and explicit future-authorization failstop; no realSGD started |

**GO recommendation is strictly for the next separately authorized reliability pilot, not permission to execute it now.** All-rejected windows, poor finite loss behavior, real-gradient instability or a failed signal remain possible future outcomes. There is no evidence here that real-data SGD has already worked.

## G. Reproducibility, verification and handoff

Compact committed evidence: [Wave4 validation](<../experiments/wave4_validation.json>); [readiness audit](<../artifacts/wave4-integration/READINESS-001.json>) verifies140priorfiles/26priorrefs/33newrefs/29criticalcodefiles. Tested implementation commit is `9de50b88e8e048a85370529ed02af04061d67ea0`; final pilot actual-byte SHA256 is `420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e`. All439tests pass in140.73seconds; `pip check` and diff checks pass. Detailed source, pairing and lifecycle reports retain exact provenance and failed/superseded run identities. Local ignored raw images/GT/models/NPZ/checkpoints/manifests are **not backed up by a clean Git commit**; transfer them with their hash-bound directory trees and byte-identical source/config snapshots. Hashes bind actual working bytes including EOL; checkout newline conversion is not silently accepted as an identical executed source. Do not overwrite or reacquire completed runs merely for a clean-looking receipt.

Read-only verification commands:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/verify_wave4_delivery.py
```

**NOT EXECUTED — only after a new explicit human request and exact finalized grant:**

```bash
.venv/Scripts/python.exe scripts/run_b2_pilot.py \
  --config configs/B2-Pilot-W4.json \
  --authorization-file artifacts/FUTURE-HUMAN-PILOT-GRANT.json \
  --ack-real-sgd \
  --output artifacts/b2-pilot/FUTURE-UNIQUE-RUN-ID
```

The future grant must bind final config/source/TRAIN/DEV/cohort/reservation hashes, tested critical code, source backbone/random-grader initialization, full-grader-only scope, one exclusivecuda0token,100updates/1600slots/900seconds and expiry. No grant exists in Wave4. Resume, if separately approved after failure/resource review, requires the exact atomic boundary and checkpoint SHA; never use a producer dispatch position.

Final integrated test count, reference check and exact clean ending HEAD are recorded in the final delivery receipt/response, avoiding an impossible self-referential commit hash inside its own committed contents. The tested implementation commit is separately recorded in compact evidence. No background training may remain at handoff.
