# HiFT Wave5 — preparation audit and blocked authorization gate

## Executive decision

**No training was authorized or executed. Execution readiness is BLOCKED, not GO.** The human request permits preparation only. PhaseA repository/data checks passed, but independent source inspection found that the frozen Wave4 launcher does not enforce several stronger mandatory Wave5 operational requirements. PhaseB has a complete **UNAPPROVED/BLOCKED draft for the current implementation**, not an approval-ready executable experiment. PhaseC must not be passed merely by setting its three human booleans to true.

- **W5-A reliability: PARTIAL / NOT EXECUTED**, not a training failure or operational success.
- **W5-B learning signal: NOT MEASURED.** No initial/final authentic-training loss, gradients or new DEV measurements exist.
- **W5-C downstream readiness: HOLD.** No new tracking-performance, causal correction, model-reproduction or R1 evidence. R1 remains on hold.
- Attempted/accepted/rejected training samples: **0/0/0**. Effective updates: **0**. Authentic-image forward/backward calls and real optimizer steps in this preparation: **0**. Pilot runtime/peak memory/checkpoint/traces are **not applicable**, not measured zero-valued training performance.
- No approved grant, GPU training lease, pilot output directory, optimizer run, retry or recovery was created. The only model construction was **CPU initialization without images/loss/optimizer**, discarded after hashing.

## A. Mandatory verification and preserved history

Starting Git HEAD exactly matched the Wave4 handoff: `b8b92a7c1a1420925dc4b243c09babac3381b66b`, with a clean working tree. Tested Wave4 implementation remains `9de50b88e8e048a85370529ed02af04061d67ea0`. Upstream remains clean at `992aa7c1605360d607af84cb804b93cd8ee81caf`. No repository reinitialization, source/config/data normalization, old reference rewriting or benchmark rerun occurred.

Commands executed:

```bash
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/wave5-preparation/PHASE-A-001/tests.xml
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/verify_wave4_delivery.py --output artifacts/wave5-preparation/PHASE-A-001/wave4-verification.json
```

Baseline: **439 passed in146.89seconds**, zero skips/failures/errors; dependencies consistent. The [Wave4 verification receipt](<../artifacts/wave5-preparation/PHASE-A-001/wave4-verification.json>) confirms **140protected historical working-byte identities,26earlier references,33Wave4 references,29critical executing files**, config and source/DEV consistency. All29critical source bytes remain unchanged. The new preparation-only code does not patch or wrap the training controller.

The [raw-asset inventory](<../artifacts/wave5-preparation/PROPOSAL-001/inventory.json>) additionally rehashes all **160original JPEGs,4original GT files,4seqinfo files and1151accepted preflight NPZ archives**,442,478,610bytes total. No arrays were regenerated. It binds the full187file conservative project/upstream source/config snapshot, temporal policies/populations/content-screen/prereg/reservations, executable and package versions. This is an inventory, **not an enforced launch-time immutable sandbox**.

Official tracker SHA256 remains `8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101`. ImageNet backbone SHA256 remains `933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8`. The released tracker was hashed only, never loaded for initialization/optimization.

CPU-only seed20261023 initialization uses the same B2 graph, strict30original ImageNet tensors plus only five missing scalar int64 zero BN counters and sourceepoch0 modes. It matches Wave4's actual untrained-forward parameter digest exactly:

| Tensor scope | Actual-byte digest over ordered names and contiguous tensor bytes |
|---|---|
| All named parameters | `52e80c8edd422bf78e0aa09ef1b9bba998f4b91ea3842863fa1c0d2225d20130` |
| Entire backbone state, including BN buffers | `c3f2d95caf15d49a922b464568e1cb6fbfa30f6922c793486b54c676e1a99e5e` |
| Random grader parameters | `f6d0b47dcbfbc706bc524e1693acf36bdf2ca8a59aa52ab6cd29fa5e9b865a36` |

The preparation process asserts CUDA was **not initialized**. This is not a trained checkpoint, authentic forward repeat or bitidentity audit of a post-training model. Preserve actual raw source/config bytes, including EOL; Git's text filters are not a substitute for the recorded byte hashes.

### Data and execution contract

TRAIN remains MOT17-02/09/10; DEV remains MOT17-05, entire known ETH/Ess collection held out.09/10 are conservatively grouped as one recording family. Identity is author within-video trajectory fragment, not global physical person. Different IDs across time/video are not presumed negatives. Positive endpoints are actual observed nonzero-gap frames; negative A/B tracks are simultaneously known distinct, and the actual augmented search crop must exclude template A with the frozen18crop-pixel+2source-pixel support margin. Unknown identities remain quarantined. No replacement/refill is permitted.

The historical1024candidate preflight remains **973accepted temporal positives,50safe negatives,1explicit crop-co-presence rejection**. It is not a Wave5 training trace or a prediction of exact1600slot outcomes. Fixed DEV128 remains64positive+64negative; its negative draws represent38unique source pairs and26repeats(max4). The cohort is a repeated weighted control, not128independent samples. Known catalog/hash/pHash alias screens and collection disjointness do not prove universal physical-person/re-edit/pretraining independence. Evaluation reservations and all frozen15sequence/2937frame UAV results remain unchanged and are not training/tuning/selection inputs.

The unchanged controller still uses indexed serial loading(`num_workers=0`,no asynchronous prefetch), the same pixel-preserving CPU NCHW bridge as the actual successful Wave4 forward, observed **global window** class/BCE/regression-weight denominators, exactly one clip/SGD per accepted window, and explicit all-rejected-window consumption. Entire grader/fusion/transformer/prediction heads are trainable; entire backbone/backbone BN are frozen. This is **not** physical-batch220/two-GPU numerical equivalence.

### Physical availability

[Captured host availability](<../artifacts/wave5-preparation/PHASE-A-001/host-availability.json>) at2026-10-23T15:52:51Z found **no Python training processes**. Device0 is RTX5070Ti,UUID `GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f`,16303MiB total/5100MiB used,44% utilization from existing non-training activity. Available host RAM was35,455,280KiB(33.81GiB) of66,810,072KiB(63.71GiB). An earlier instantaneous sample was only703,912KiB(0.67GiB) available, so headroom is not stable. No application was killed or modified. An instantaneous observation is not an exclusive lease; availability must be rechecked at any later authorized launch.

## B. Exact draft specification and PhaseC disposition

[Complete UNAPPROVED proposal](<../artifacts/wave5-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json>) SHA256:
`2a395299db922658678a0f07eeca1b6bfeb9b66414fb3093132be8ed2b662dc9`

[Full source/data/model-initialization inventory](<../artifacts/wave5-preparation/PROPOSAL-001/inventory.json>) SHA256:
`3ff399383ebaa6d57a47a602714bdf3c8f91aaa683d94fd661ae301e11fd190f`

The payload binds the union of every named path/hash under those maps, not permission to substitute a wildcard source tree. It is an immutable local preparation artifact, not a human signature, authorization receipt, pilot manifest or guarantee of hermetic import resolution.

| Proposed item | Frozen value / explicit status |
|---|---|
| Experiment identifier | `B2-W5-RELIABILITY-20261024-001` (no existing output/grant at preparation) |
| Config | [B2-Pilot-W4](<../configs/B2-Pilot-W4.json>),byte SHA256 `420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e` |
| Manifest | `2660a96d72ed1645b822120d81835f7a66f93b65da9a4648f6a16fcc04bdba54` |
| TRAIN policy | `c0d811b945f385084daa8ed27042e2bc0178a4859ffab34868fda8304d33afa9` |
| DEV policy | `15bf57bc1b7107cf5d6ca51ed72a54c520e13fda7080a2014c833bb2afcd6a36` |
| Fixed DEV cohort | `eae252a734198935c62a80fdbd8239d05169c710a7182d6188de2d8d21c1efeb` |
| Reservations | `a4f0b91e0c94181f4f8dbc09a09762fa722d97cead728bd553c333447f508e7c` |
| Hardware / seed | Concrete above GPU UUID,cuda:0 exclusively;TRAIN20261023,DEV20261024,float32,TF32off |
| Scope | Entire random-initialized corrected grader; pretrained BB+BBBN completely frozen; sourceepoch0 |
| Optimizer | SGD,graderLR.0025,momentum.9,WD.0001,clipnorm10; no accelerated schedule |
| Batch/accounting |4attempt slots/physical micro,16attempt slots/window,≤4micros,no refill; allreject consumes attempts without step |
| Hard limits |100effectiveupdates,1600indexedattemptslots,100windows,900cumulative seconds; exhausted attempts before100updates=completionFAIL |
| Monitoring | Fixed DEV0/25/50/100; actualrunDEV0 baseline; fixedlast100 primary;bestDEV earliesttie secondary cannot rescue |
| Signal | Update100DEV≤.95×update0DEV; distinct from reliability/3×nondivergence; no tracking accuracy claim |
| Retry/recovery | One fresh invocation only; no automaticretry/resume/scope expansion; any failure requires renewed exact checkpoint/remaining-resource review |
| Proposed expiry |2026-10-24T16:00:00+00:00 = **2026-10-25 00:00:00 Asia/Singapore** |
| Additional proposed memory guards | Hostprocess8GiBmaxRSS,availablehost≥4GiB,CUDAreserved≤.8devicecapacity; **unapproved proposals, not implemented existing guards** |
| Human state | `human_approved=false,separate_post_wave4_request=false,allow_real_sgd=false`; humanreference and GPUtoken are null |
| Execution state | `execution_ready=false`; **NOT ELIGIBLE FOR TRAINING APPROVAL** until mandatory executable gaps are repaired and newly frozen |

Exact old-CLI argv is bound in the draft. Shown for audit only—**do not execute**:

```bash
.venv/Scripts/python.exe scripts/run_b2_pilot.py \
  --config configs/B2-Pilot-W4.json \
  --authorization-file artifacts/wave5-authorization/B2-W5-RELIABILITY-20261024-001.approved.json \
  --ack-real-sgd \
  --output artifacts/b2-pilot/B2-W5-RELIABILITY-20261024-001
```

The approved file above does not exist. Existing gate must reject the false draft flags; no training launcher was invoked to test that. Do not flip the historical config authorization metadata: the frozen validator expects Wave4's false flags. Later approval must be a distinct new artifact referencing the final reviewed proposal, not mutation of this draft or a self-generated human reference. No approval is being solicited for execution of this blocked command.

## C. Independent execution/authorization audit — mandatory launch blockers

Two independent read-only subagents inspected actual executing source and the exact frozen bindings. Neither edited files, initialized GPU, performed experiments or created a grant. Both concluded **the existing launcher alone cannot meet the exact new Wave5 mandate**. This is a narrower operational finding, not a refutation of the measured Wave4 mathematics, forward path or synthetic boundary replay.

1. **W5-B1: inclusive cumulative deadline.** [Launcher:117–168](<../src/hift_research/training/wave4_pilot.py#L117-L168>) starts the clock after grant/data/output setup, detects DEV/checkpoint overrun after work and lacks a hard external watchdog. Resume's snapshot omits later save/failed/postboundary time. [Config:149](<../configs/B2-Pilot-W4.json#L149>) explicitly documents external/manual resource review, not implemented strict accounting. Required remedy: external monotonic ledger/watchdog bounded by900cumulative seconds and grant expiry, counting initialization/I/O/checkpoint/failedwork; fresh-only one-shot pilot.
2. **W5-B2: grant binding and actual lease.** [Gate:30–67](<../src/hift_research/training/wave4_pilot.py#L30-L67>) checks config/three data hashes/budgets/expiry and truthy human reference/token, but not exact experiment ID,argv/output,one-use consumption,all29frozen sources or physical GPU ownership. Fresh controller snapshots current training sources rather than enforcing the whole previous snapshot. Temporal policy correctly enforces its20source hashes; that does not bind the entire execution closure,including AST-executed upstream loss code. Required remedy: new trustworthy exact-human approval envelope, immutable import/source/environment closure and atomic one-use launch/GPU lease before importing mutable execution code. No cryptographic human signature was established by the legacy boolean mechanism.
3. **W5-B3: immediate gradient/memory stops.** [Engine:119–148](<../src/hift_research/training/wave4_pilot_engine.py#L119-L148>) audits gradients after the complete window; CUDAreserved safety is checked after a step; host safety limits are absent. Required remedy: per-micro finite checks and precommit CPU/GPU memory stops, with exact proposed numerical caps reviewed before authorization.
4. **W5-B4: fatal contract versus benign rejection.** [Pipeline:195–229](<../src/hift_research/temporal_contract/pipeline.py#L195-L229>) catches shape/dtype/nonfinite/mask ValueErrors with geometry rejections. Required remedy: preserve allowed geometric no-support/co-presence rejection/no-refill semantics but escalate invalid input/identity/mask/hash faults before any effective-window commit. Do not rewrite the frozen sampler recipe or silently redefine the population.
5. **W5-B5: live invariants.** Frozen requires_grad/modes establish intended freezing, not an observed postupdate bitidentity audit. Required remedy: verify backbone parameter+BN bytes, active parameter/optimizer scope and source/data/checkpoint hash drift through execution; record before/after state and stop before publishing invalid state.
6. **W5-B6: durable evidence.** [Engine:154–166](<../src/hift_research/training/wave4_pilot_engine.py#L154-L166>) stores total loss/counts/norm and CUDA allocator timings, not required loss components,global denominators,class/reject breakdown or host memory. [Failure writer:192–196](<../src/hift_research/training/wave4_pilot.py#L192-L196>) omits last checkpoint/measurements and early initialization faults precede its try block. Required remedy: append-only flushed telemetry plus exact last committed update/attempts/checkpoint/failure-window/cumulative-time/remaining-budget evidence; postprocess separate reliability versus `.95`fixedlast signal without changing losses or checkpoint selection.
7. **W5-B7: physical availability.** Capture is not an actual exclusive lease; GPU is already active and host headroom varies substantially. Recheck and acquire ownership immediately before a later approved run; do not kill unrelated processes, assume a truthy token means ownership, or start a competing GPU job.

**No runtime safety/telemetry adapter was implemented in this preparation handoff.** The new [inventory/draft helper](<../src/hift_research/wave5_preparation.py>), [preparation CLI](<../scripts/prepare_wave5.py>) and [18unit tests](<../tests/test_wave5_preparation.py>) prove only preparation/byte-inventory/no-overwrite/false-approval behavior. They do not solve the above executable gaps or authorize training. Adding the necessary adapter changes executable inventory/likely invocation and therefore requires a **new complete freeze and exact human review**, even though the29Wave4files/config/data remain immutable.

## D. Measurements and scientific interpretation

All requested training measurements—initial/final training loss, DEV0/25/50/100, gradient traces, posttraining backbone/BN/optimizer/RNG audit, pilot runtime/memory and checkpoint SHA—are **NOT MEASURED/NOT CREATED**. Historical forward-only DEV3.4531175792217255 remains historical and is not substituted for a new pilot baseline. Preparation itself took41.50seconds for CPU initialization and source/data hashing, not training throughput; preparation-memory peaks were not measured. No pilot failure, last committed update or recovery checkpoint exists; the stop reason is **preauthorization execution-readiness block**. All1600attempts/100updates/900seconds remain theoretically unspent, but this is not available authority or a carryover grant.

Demonstrated: prior evidence/assets remain intact; exact random initialization matches the successful Wave4 forward; the current recipe/path is unchanged; independently audited launch gaps are real and a false draft cannot be represented as approved. Not demonstrated: real-data training reliability, learning, UAV generalization, tracking performance, causal correction benefit, convergence or novelty.

MOT17 person imagery versus UAV aerial targets, tiny source cohort, frozen backbone, dependent DEV controls, conditioned safe-negative prevalence and short temporal gaps remain limitations of the eventual pilot. No result exists to diagnose underfit/overfit or gradient stability. There is no new evidence supporting R1.

## E. Exactly one next bounded experiment and prerequisites

Recommend **only the same first100update corrected-B2 grader-only reliability pilot**, not another architecture/seed/LR/benchmark experiment. Its information value is whether authentic observed-global-denominator optimization is finite, state-safe and completes with exact no-refill accounting, with an exploratory fixedlast DEV-loss signal considered separately.

Prerequisites before requesting its real-SGD approval: implement and independently CPU/synthetic-validate an additive W5 deadline/lease/authorization/invariant/telemetry guard outside the29frozen files; preserve all recipe/data bytes; acquire safe host/GPU availability; resolve every new executable/hash/command; run full regressions and the historical verifier; issue a **new** fully bound approval-ready proposal with exact expiry and memory thresholds. Obtain explicit human approval to that exact new proposal. If any future window fails, retain evidence and stop without retry; any recovery needs new human approval and cumulative failed-work accounting. Do not train, rerun UAV benchmarks, unfreeze the backbone, select on DEV or implement R1 in the meantime.

## F. Integration and handoff

Baseline439tests and new18preparation tests pass. **Final integrated457tests passed in195.02seconds**, zero failures/errors/skips; `pip check` and diff checks passed. Final test and preservation receipts are recorded in [compact preparation evidence](<../experiments/wave5_preparation.json>) and the final handoff response. Raw inventory/proposal/artifact files remain local ignored evidence, not Git backup. Only additive preparation source/tests, compact evidence and documentation are committed. No historical experiment/config or checkpoint is altered. Exact final Git HEAD belongs in the final response/ignored ending receipt, not its self-referential committed report.
