# Wave6 independent source and gate review

## Disposition — reviewed engineering signed off; execution BLOCKED

Task-19, reviewer `literature`, 2026-10-24 Asia/Singapore. **This is an engineering review, not authority to run authentic SGD.** G5 remains genuinely **BLOCKED** by unresolved physical GPU ownership; G6 is dependent-blocked. No new real pilot proposal, human ACK, grant, authentic training, CUDA experiment or benchmark is recommended by this review.

The reviewed implementation closes important Wave5 operational gaps through an additive wrapper around the unchanged Wave4 core. All six specific source findings below have been reread in their corrected form, including the final interpreter/environment fix. Final post-fix evidence reports **626 passed, zero failures/errors/skips**; the XML header was independently read and the receipt hashed. No additional mandatory source defect remains open in this review. This signs off the stated source/native/CPU engineering evidence, **not overall execution readiness**: the final native availability receipt still has 38 unresolved WDDM `C+G` processes. G5 is the sole current operational blocker; G6 is dependent-blocked. The final RAM snapshot passes the proposed threshold and is not the current blocker.

## Scope and independently checked evidence

- Read the Wave6 authorization, bootstrap, entry, worker, adapter, runtime, telemetry, ledger, supervisor and native Windows Job/lease implementations, plus all five Wave6 test modules and their added integration cases. Review is cross-file/source-based; I did not rerun tests, import Torch, construct a model, open authentic images, perform backward, create grants, acquire a GPU lease, kill processes or commit.
- The only executions by this reviewer were small read-only `sha256sum` and Git identity/status commands. All **29** executing files in the historical [critical-source reference](<F:/Code/HiFT/artifacts/wave5-preparation/PHASE-A-001/wave4-verification.json#L6-L36>) match their exact SHA256 values. This includes the original loss, graph, optimizer/controller, layout bridge and immutable upstream sources. This is raw-byte comparison, not a Git text/EOL approximation.
- Repository HEAD: `1a789938eb5afbf63f1de64d2ae437ee39f29772`. Upstream HEAD: `992aa7c1605360d607af84cb804b93cd8ee81caf`; upstream working tree was clean.
- Independently rehashed [frozen pilot configuration](<F:/Code/HiFT/configs/B2-Pilot-W4.json>): `420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e`.
- Independently rehashed historical [UNAPPROVED Wave5 draft](<F:/Code/HiFT/artifacts/wave5-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json>): `2a395299db922658678a0f07eeca1b6bfeb9b66414fb3093132be8ed2b662dc9`; historical [inventory](<F:/Code/HiFT/artifacts/wave5-preparation/PROPOSAL-001/inventory.json>): `3ff399383ebaa6d57a47a602714bdf3c8f91aaa683d94fd661ae301e11fd190f`. Neither was converted into approval.
- Read [CLI-004 JUnit evidence](<F:/Code/HiFT/artifacts/wave6-verification/INTEGRATION-CLI-004.xml>): **5 tests, 0 failures/errors/skips, 44.176 seconds**. This is an inspected Lead-produced result, not a reviewer rerun. It covers actual native CPU/random-data CLI success, exact checkpoint/reuse checks, a stopped post-backward hang, checkpoint-rehash failure finalization, checked-source import and synthetic-mode separation.
- Read [FINAL-002 JUnit header](<F:/Code/HiFT/artifacts/wave6-verification/FINAL-002.xml>): **626 tests, 0 failures, 0 errors, 0 skips, 196.838 seconds** in JUnit (Lead's CLI summary: 196.85 seconds). This replaces the 620-test pre-final result; 169 new tests are additional to the 457-test Wave6 starting baseline, not additional to the older 439-test Wave4 report.
- Read [HISTORY-002](<F:/Code/HiFT/artifacts/wave6-verification/HISTORY-002.json>): PASS for 140 protected working-byte identities, 26 older and 33 Wave4 references, 29 critical source hashes and the frozen config. Its embedded `integrated_tests=439` and `GO_FOR_SEPARATELY_AUTHORIZED_BOUNDED_GRADER_ONLY_PILOT` recommendation are **historical Wave4 fields**, not the current test count or a Wave6 execution GO. Lead collected the final managed job with exit 0 and reported `pip check` with no broken requirements; I did not rerun pip/tests.
- Read and independently hashed the final [postquiescent physical receipt](<F:/Code/HiFT/artifacts/wave6-lease-validation/physical-availability-002.json>): 2026-10-23 17:39:45.162679 UTC / 2026-10-24 01:39:45.162679 Asia/Singapore. Availability remains false solely for 38 unresolved WDDM `C+G` processes. RAM available is 17,410,555,904 bytes (16.21484375 GiB), with 13,665,042,432 bytes GPU-free of 17,094,934,528 total. Receipt states no GPU compute, no held lease and zero real SGD steps. The absolute System32 SMI producer SHA is `3fd0d31eaa346704ec7fa6872c2009898d30223dd69e89573eb7f12cfb521e05`.

## Specific findings and their disposition

### R1 — checkpoint rehash exception could leave a false COMPLETED ledger: corrected

Originally, checkpoint revalidation could raise after the supervisor had returned `COMPLETED`. The outer exception handler recorded failure, but finalization selected the unchanged success status. CLI failure alone did not make that authoritative ledger truthful.

The reviewed [entry exception handler](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L98-L104>) now changes an existing result to `FAIL_ENTRY` before finalization. The metadata-only injected-I/O regression is included in CLI-004. Checkpoint byte mismatch is independently converted to `FAIL_CHECKPOINT`; neither a zero child exit nor a previously successful supervisor receipt suffices for success.

### R2 — hashed source did not guarantee imported source: corrected within the stated project boundary

Ordinary timestamp-valid project bytecode can be stale after same-second/same-size edits. `PYTHONDONTWRITEBYTECODE` alone does not stop bytecode reads. Project native/sourceless shadow modules and sibling standard-library shadows were also outside the original guarantee.

The final reviewed [bootstrap isolation and source loader](<F:/Code/HiFT/scripts/run_b2_guarded.py#L7-L77>) removes project/script/PYTHONPATH search locations before source-module stdlib imports, rejects project aliases and native/sourceless executable shadows, and patches `SourceFileLoader.get_code` for only the three approved project roots. It **reads source bytes once, compares those bytes to the approved SHA256 map, and compiles those same bytes**. Normal imports and source-file importlib specs therefore do not execute stale project bytecode or reopen different bytes after a successful check. Unlisted project sources fail. Installation precedes the first project import in both parent and worker. The [worker](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L37-L50>) checks the actual loader-class fingerprint, scope marker and no-write policy before Torch.

The initial global fresh-cache approach was not retained: it unnecessarily recompiled trusted Torch dependencies and produced a genuine 30-second synthetic-fixture deadline failure. That failed observation is not a PASS. The replacement preserves the same attempt slots, losses, caps and input policy. Trusted interpreter/site startup and installed dependency caches remain an explicit trust boundary; this is not a complete hermetic Python environment or an administrator-resistant sandbox.

### R3 — experiment identifier could escape the bound ledger directory: corrected

An absolute or traversal-containing experiment ID formerly became a ledger pathname despite synthetic fixture path confinement. [Identifier validation](<F:/Code/HiFT/src/hift_research/wave6/authorization.py#L88-L95>) now permits bounded safe basenames and rejects Windows devices, with validation of both identifiers. [Entry](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L62-L70>) additionally resolves the ledger target and requires its parent to equal the exact claim directory **before** directory/file creation. This closes a concrete path-confinement defect, not merely an adversarial-code scenario.

### R4 — uncertain native lease release erased its blocking evidence: corrected

Originally the persistent artifact was removed before checking `ReleaseMutex`. An API failure could therefore lose the fail-closed artifact even though the parent reported cleanup failure.

The reviewed [release sequence](<F:/Code/HiFT/src/hift_research/wave6/lease.py#L279-L305>) retains the artifact until native release and handle-close success, rechecks its unchanged identity before unlinking, marks uncertainty on any failure, and forbids automatic retry. Entry releases only after the supervisor proves the descendant tree empty; unproved containment retains the artifact for manual reconciliation. This does not grant permission to remove another owner's record or terminate unrelated processes.

### R5 — synthetic flag must never select authentic execution: corrected

[Entry](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L57-L65>) now requires the requested synthetic mode to equal verified envelope scope before reserving a ledger. A synthetic-looking command cannot dispatch authentic data even with a mocked otherwise-valid capability. CLI-004 contains the cheap metadata-only regression; it creates no real human grant.

### R6 — Python flags/environment could evade reconstructed exact argv: corrected

The first source freeze reconstructed command identity without interpreter switches, and inherited `PYTHONOPTIMIZE` could strip assertions when approved source was compiled. Normalizing script spelling also risked representing a different literal command as identical. In addition, merely deleting keys from a supplied environment dictionary was ineffective because the native process wrapper merges overrides onto `os.environ`.

The final [builtin-stage bootstrap checks](<F:/Code/HiFT/scripts/run_b2_guarded.py#L7-L19>) reject optimization and any interpreter argv outside the approved direct-script invocation before ordinary stdlib source imports. [Actual command binding](<F:/Code/HiFT/scripts/run_b2_guarded.py#L113-L121>) preserves raw script/argument spelling. The [worker profile](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L20-L35>) binds optimization zero, `PYTHONOPTIMIZE='0'` and `PYTHONHOME=''`; [entry passes those explicit values](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L91-L94>), so native environment merging cannot reintroduce the removed settings. [Worker verification](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L36-L51>) checks the actual flags/environment before Torch. Read [INTERPRETER-001 JUnit evidence](<F:/Code/HiFT/artifacts/wave6-verification/INTERPRETER-001.xml>): six tests pass with no failures/errors/skips in 4.457 seconds, including an actual native base-executable/venv child launched from contaminated parent environment. Their source was independently read; this was not a reviewer rerun. These tests require no real human grant or Torch execution. The final 626-test suite subsequently passed.

## Seven safeguard call-chain assessment

1. **B1 — inclusive time and containment.** [Bootstrap](<F:/Code/HiFT/scripts/run_b2_guarded.py#L7-L34>) anchors time before normal preparation and bounds hashing. The consumed [ledger](<F:/Code/HiFT/src/hift_research/wave6/ledger.py#L86-L164>) uses the original monotonic start, UTC expiry and no-reopen one-shot reservation. Parent hard-exit/emergency protection remains active during preflight, cleanup and finalization; failure to finalize does not silently disarm it. [Windows Job creation](<F:/Code/HiFT/src/hift_research/wave6/windows_job.py#L101-L174>) atomically assigns a suspended child before first instruction, with noninherited KILL_ON_JOB_CLOSE and restricted inherited stdio handles. The external parent, not training-loop cooperation alone, bounds a hung backward. Cleanup margins do not extend the approved budget. Missing/corrupt final evidence is failure/uncertainty, not success.
2. **B2 — exact permission and scope.** [Authorization](<F:/Code/HiFT/src/hift_research/wave6/authorization.py>) checks exact proposal/ACK schemas and full bytes/digests, user-role message/session reference, exact ACK text, command/cwd/output, expiry, source/data/init maps, one-use scope and fixed caps. Atomic claim precedes model/optimizer construction and remains consumed after later failure. [Dispatch](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L24-L63>) checks inherited capability, bound inputs, physical/logical interpreter identity, selected environment, actual direct-parent PID/creation claim, and physical lease nonce/owner—not an arbitrary ancestor. The reviewed authentic bridge is scoped to this verified worker; booleans/tokens alone are not approval. R6 is corrected in source and included in the final passing suite.
3. **B3 — immediate runtime stops.** [Adapter hooks](<F:/Code/HiFT/src/hift_research/wave6/adapter.py#L39-L59>) guard after every backward, before clipping, before SGD and after SGD; [runtime checks](<F:/Code/HiFT/src/hift_research/wave6/runtime.py#L154-L220>) inspect finite parameters/buffers/momentum/gradients and finite aggregate norms, clipped norm, native peak RSS ≤8 GiB, available host RAM ≥4 GiB and reserved CUDA fraction ≤0.8. Authentic execution uses native probes, never synthetic counterfactual resource fixtures. These checks add observation/rejection; the frozen controller still owns loss/backward/clip/SGD.
4. **B4 — geometric rejection versus fatal input.** [Typed rejection and batch validation](<F:/Code/HiFT/src/hift_research/wave6/runtime.py#L222-L310>) admits the two named genuine geometry outcomes with supporting evidence, while invalid order/status/seed/identity/shape/dtype/device/finite/mask/hash/co-presence conditions are fatal. Unknown pipeline ValueErrors cannot silently become allowed no-refill rejections. Attempt slots and the original sampler distribution are not refilled or replaced.
5. **B5 — live invariants.** Runtime verifies full backbone/BN state bytes, buffer schema, parameter identity/trainability/modes, ordered optimizer scope/options, epoch-0 learning rates and allowed momentum state. Entire corrected grader/fusion/transformer/prediction heads are active; entire pretrained backbone and backbone BN remain frozen. [Authentic initialization](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L134-L162>) checks the physical CUDA UUID and all three expected initialization digests before constructing the optimizer. Runtime source/data drift checks supplement initial full asset binding; historical NPZ archives are not substituted for live input arrays.
6. **B6 — durable and truthful accounting.** [Journal](<F:/Code/HiFT/src/hift_research/wave6/telemetry.py>) fsyncs hash-chained detached primitive records. [Adapter](<F:/Code/HiFT/src/hift_research/wave6/adapter.py#L69-L137>) logs planned attempts/classes/reasons, observed denominators, loss components, gradients, optimizer transitions and checkpoints. A post-step or publication failure taints live state, permits no continuation/checkpoint, and retains the last published boundary. Parent summary distinguishes planned versus observed failed work and treats torn/corrupt evidence conservatively. Checkpoint bytes are rehashed; [entry finalization](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L98-L132>) follows lease cleanup and receipt publication. CLI success also requires final ledger/timing, tree-empty/job-close and protocol checks, not just exit 0.
7. **B7 — actual physical availability and exclusive lease.** The native [lease implementation](<F:/Code/HiFT/src/hift_research/wave6/lease.py>) binds GPU UUID, actual PID plus creation time, authorization digest and nonce, with a shared Windows mutex and persistent artifact. Existing/stale/abandoned/uncertain ownership fails closed. Entry rechecks native availability and holds ownership through descendant cleanup. SMI comes from the absolute native System32 producer with producer identity recorded; names such as desktop or GUI processes do not waive `C+G` ambiguity. Implementation exists, but **the actual current physical prerequisite does not PASS**.

## Scope and gate honesty

- The final independently read native receipt reports **38** WDDM processes as `C+G`, preventing unambiguous absence of foreign compute. The earlier count of 36 is superseded. I did not rerun SMI or reinterpret executable names. This is the sole current external operational G5 blocker, not a failed unit test and not permission to kill other workloads.
- Earlier RAM samples were volatile (roughly 55 MB, then 5.17 GiB), but the final postquiescent sample is **16.21484375 GiB available**, above 4 GiB. **RAM is not a current blocker in that receipt.** An instantaneous pass is not sustained headroom or GPU ownership proof; runtime guards must remain active.
- **G6 remains blocked by G5**, despite the final 626-test engineering PASS. Passing CPU/random-data and native process tests cannot turn the overall decision into TECHNICALLY READY while G5 is false.
- CPU bitexact/model-optimizer/RNG/replay evidence supports additive-hook correctness for tested CPU cases. It does not independently prove Wave6 CUDA bitidentity, authentic-data reliability, successful 100-update completion, learning or tracking accuracy. Fixed-LAST DEV ≤0.95×initial is distinct from operational ≤3× nondivergence; descriptive best-DEV cannot rescue failed fixed-LAST signal.
- Any eventual scope remains **at most** 100 effective updates, 1600 fixed indexed training attempt slots, 100 windows, 900 inclusive seconds, physical four attempts/micro and sixteen/window, epoch 0, grader LR .0025, no refill/retry/resume. All-rejected windows can prevent reaching 100 updates; that is completion failure, not grounds for expansion.
- The frozen legacy launcher remains historical code. Calling the new route the sole **authoritative supported production entry** is a policy/scope statement, not a claim that arbitrary Python or historical entry files are technically sandboxed. Human message references are trusted transport attestations, not cryptographic signatures.
- No freeze/proposal generator was required or issued while G5 is blocked. A source engineering snapshot/report is not a new executable pilot proposal, and the historical Wave5 draft/expiry must not be reused.

## Smallest next action

The final source/tests/history/availability evidence has now been collected and reviewed; keep overall status BLOCKED. The smallest remaining operational action is to establish an unambiguous non-competing physical GPU ownership/availability state, without terminating unrelated work or waiving `C+G` by process names, then recheck host headroom at any later authorized launch. Only after every required gate passes may a newly frozen proposal with a new expiry be prepared for a separate genuine human decision. This report grants none of that authority.

## Independently fingerprinted final source snapshot

These raw hashes identify the final small-fix source/test snapshot reread by this reviewer. They are engineering evidence, not a launch inventory, proposal or ACK. No implementation file was edited by this reviewer.

```text
fd4f73985fc0e4312b45b7919bcdb79ffba19b5bff358141f26e6adb73e8ac86  scripts/run_b2_guarded.py
97fbb60ed768167352ca633f724c9f10ff7a76dab426b97c8944fae06be06232  src/hift_research/wave6/__init__.py
2a9231f06f91489e9e91238f3cf224638c626747b8413a9702285668a6b49504  src/hift_research/wave6/adapter.py
6ff8f7be6b448c6b18dbf0ed433939c3039316f125b5a496ac6e4a762595d248  src/hift_research/wave6/authorization.py
037d04f20becd04595fbe2a6ea398c7a922e17a25f7b972220b4fdd2ea8bfcb0  src/hift_research/wave6/entry.py
376c48ede95f37b9db985e4d82f360e4381b619a290918af36baa4fefa59019c  src/hift_research/wave6/lease.py
49ef6a90757aa0eb2d2c2b20eda0c17c0774878dc2bafbebb4cd471e05d17560  src/hift_research/wave6/ledger.py
51ac230e25d4927d88464f1dcaa2e15e70680fbb13602dc9db281e40d88ff3aa  src/hift_research/wave6/runtime.py
ccf45f7f9e09bfd6553b635c76b2bd0368ac4465ae1a24b139769600a4751631  src/hift_research/wave6/supervisor.py
72a066357a226aa8a4a82f76fb21efcd2bd67a91379909144c6542068d1975d3  src/hift_research/wave6/telemetry.py
85cffb16bbd229ea26668f555fd41a57cbe13dcc64aeb7687aeafef0df77ab7d  src/hift_research/wave6/windows_job.py
37858264feb3ecb044db14b80c4b210b28a6178e439c567e85a71d26bff46508  src/hift_research/wave6/worker.py
7cea04d15e159cec4c5cab1e28e9b8431c897325e4a57d50de87454c95bf1a3c  tests/test_wave6_authorization.py
0578a23ff02ac6af44c46a3236606d0977c31375f99b4365070e8b33fc79baa6  tests/test_wave6_integration.py
81d02d2b42c1599173ba140dc402c41e8278cbde968eed8e05ea8ce87dfb3bfd  tests/test_wave6_lease.py
2fce7194717a9635df002bae0f86247584b5179e54185b83c5e7fb16fe500bdf  tests/test_wave6_runtime.py
230412d77bbee73be1457a831e1d528adb2de0a58cb5cb22b70ffb7a04a96a77  tests/test_wave6_supervision.py
```

Independently checked receipt hashes:

```text
8c23cd235c46940e60b6a03a1eca19baa216e50f6f87735e6c380d3c34d2cb35  artifacts/wave6-verification/FINAL-002.xml
6a9efcbcc9391b2287b1c16a63b29a5aaca8bb4535ff161d182db6a790b8bd57  artifacts/wave6-verification/HISTORY-002.json
fa3aaf0cc077f32e6017d6a38a8fa0e3a76e08568dc0cd0c279fdd88684dd5a9  artifacts/wave6-verification/INTERPRETER-001.xml
e430d6eb580041d614aa009c493d9aa0fa23bace8cb95d4ba1361fd6e95546d6  artifacts/wave6-verification/INTEGRATION-CLI-004.xml
c92347e6798d7e34aaec8add4efac567eb69b8eba8a5872a177f909b57f206bc  artifacts/wave6-lease-validation/physical-availability-002.json
eabf19773531e5fd035e36cbbddb2ec7dac7538a00c94b305149df5babc13575  artifacts/wave6-lease-validation/native-smi-002.xml
```
