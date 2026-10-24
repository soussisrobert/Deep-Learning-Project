# POLICY B ENGINEERING READY — HUMAN APPROVAL REQUIRED

## Operational decision

- **Policy A — STRICT_PHYSICAL_EXCLUSIVITY** remains the default when no explicit policy binding is supplied. Its original context-classification gate is unchanged. Opaque WDDM C+G clients still block A; they are neither counted as compute jobs nor declared graphics-only.
- **Policy B — COOPERATIVE_GUARDED_GPU** is separately named/versioned and hashes its exact specification and implementing source map. Only the supported participating HiFT guarded invocations share the atomic host-global UUID mutex/durable lease. This is **not hardware-wide exclusivity**, universal process attribution, or protection against deliberately invoking unsupported legacy/arbitrary Python code. Unrelated GPU clients may remain present under explicit human acceptance and trustworthy resource measurements.
- This wave does not investigate another context-classification API. The added NVML adapter reads documented global memory/UUID only and does not create a CUDA context or infer context absence.

## Acceptance criteria (planned controls, not training measurements)

| Observation | Exact criterion / frequency | Rationale / failure behavior |
|---|---|---|
| Physical GPU free memory before launch | >= 4 GiB, native snapshot and first suspended-worker sample | >12x recorded synthetic-training reserved peak316MiB; >21x authentic-forward reserved194MiB. Those profiles exclude full context/driver residency and do not measure authentic backward requirements. |
| Runtime physical GPU free memory | >= 2 GiB, sample every1s and existing safe boundaries | Headroom is remeasured, never presumed from history. |
| Available host RAM | >=4GiB | Native physical total/available consistency required. |
| Training-process peak RSS | <=8GiB | Parent measures actual child PID; worker also measures its own peak working set. |
| CUDA allocator reserved | <=80% physical total; allocated<=reserved | Worker heartbeat every0.5s plus safe-boundary checks; parent compares physical total/UUID. |
| Native resource/authority query | <=2s | Independent50ms watchdog kills the existing controlled Job tree on a blocked sample. |
| Maximum trustworthy observation age | <5s | Missing/stale/future/rolled-back/invalid readings stop, not substitute zeros. |
| Initialization-only allocator gap | At most30s from monitor start | Explicit pre-CUDA unknown state, not a zero measurement. Native memory, UUID and lease/claim checks are never waived. CUDA telemetry must activate before model construction. |
| UUID, consumed claim and lease | Every1s and every existing safe boundary | Exact physical UUID, actual parent PID/creation time/nonce, policy/authorization digest and expiry. |
| External processes | Native SMI snapshots at launch/acquisition and release | Retain raw PID/type/name/used-memory; names are observational, never graphics-only evidence. Continuous NVML global memory is not process attribution. |

Any insufficient resource, noninteger/out-of-range/inconsistent native byte counter, missing or wrong UUID, changed lease/claim/source/input, nonfinite gradient or invariant violation stops without automatic retry. The independent external inclusive900s deadline and Job containment remain authoritative. Measurements are instantaneous bounded safeguards, not continuous physical reservation.

## Minimal implementation

The scientific model, optimizer, loss, sampler, TRAIN/DEV cohort, datasets, initialization and configuration remain unchanged. Changes are confined to the existing guarded infrastructure, one explicit policy module, one stdlib monitor adapter, preparation tooling and tests. Existing per-micro gradient checks, precommit checks, hash-chained evidence, one-use claims and durable failure accounting are reused. Cooperative checkpoint bytes remain unvalidated staging evidence until post-save safety passes; failed windows/tainted live state cannot publish successful checkpoint events. Old artifacts remain intact; inherited synthetic CLI helpers append new metadata case directories rather than rewriting old cases.

Two separate exact human decisions are mandatory. The acknowledgment is literally those two decision lines joined by a newline; there is no third hidden approval. Complete proposal SHA256 binds command/output/expiry/scope/resources/source/data/initialization; the operational-policy SHA binds the distinct policy specification and implementation. Neither historical grants nor the current engineering mandate authorizes training.

## Verification and frozen proposal

Executing implementation commit: `48a315e8921a81d5f2e44d5d380340ed78651cb0`, directly descended from Wave6 `caa23e2a9e11fef4a553b2a06262733b7b04affb`. Subsequent handoff/bookkeeping changes do not change the frozen executing source inventory.

- [Complete regression receipt](<../artifacts/wave7-verification/REGRESSION-001.json>): **880 PASS, zero failures/errors/skips**, 303.231s JUnit /303.25s console;139 tests beyond the741-test starting suite. Source was held quiescent across the final run. The244-file inventory is total coverage, not244 new files.
- [Independent review](<wave7_independent_review.md>): no unresolved mandatory engineering finding; all15 required cases have an explicitly leveled coverage matrix (validators, actual native mutex/Job, actual B-signed synthetic CPU CLI and inherited shared safeguards). Intermediate fail-closed source-drift/Windows-sharing failures and their repairs are disclosed, not counted as acceptance runs.
- [Post-freeze audit](<../artifacts/wave7-verification/FROZEN-AUDIT-001.json>):244 source/config/test files,1332 data/reference files (including1319 authentic assets),3 initialization input files,8 bindings and2532 existing historical evidence files rehashed;29 frozen scientific files and187 Wave5 source files unchanged. All3 seeded tensor initialization digests match.160 raw source frames and1151 accepted archives retain identity. Upstream `992aa7c1605360d607af84cb804b93cd8ee81caf` is clean. Original pilot configuration SHA256 remains `420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e`. Dependency check reports no broken requirements.
- [Known-process/lease observation](<../artifacts/wave7-verification/KNOWN-PROCESSES-001.json>):13 observed Python-family processes, no unavailable command lines and no named HiFT training-script candidates. This is not universal attribution or permission. Preparation separately rechecked the actual shared UUID mutex and production artifact namespace immediately before publication, then released all transient ownership.
- [Fresh freeze review](<../artifacts/wave7-preparation/PROPOSAL-001/preparation-review.json>): at2026-10-24T10:44:06.691005Z, GPU free7,422,869,504B (~6.91GiB), host RAM available20,915,077,120B (~19.48GiB),40 raw C+G clients, PolicyB eligible. They remain opaque, neither40 proven compute jobs nor40 proven graphics-only clients. These measurements are not a future reservation. The [same earlier live XML under A and B](<../artifacts/wave7-verification/READINESS-001.json>) blocks A and passes B. Read-only NVML adapter checks imported no Torch or CUDA context.
- The [new immutable proposal](<../artifacts/wave7-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json>) has SHA256 **`45d04e2d6738cdd4a40f6e74c0a1db5c159652d8977a14c246ac4e506982e7a5`**, canonical digest `fdbee79cfadbb0626e91b3da4728f631eab4a30c8f7e0478eabe94691c5f23eb`, and expiry **2026-10-25T10:43:42.219084Z**. Policy SHA256: `8fb121701a563014b1f7fa8811c8e7fa86fc69feb616242c807743e98dd293f0`; exact implementation-map SHA256: `acdd58a80b17e62a0b2e8cada6300f4625f33534e7fcef080e169cb4511e70ad`.
- [Human review request](<../artifacts/wave7-preparation/PROPOSAL-001/HUMAN-REVIEW-REQUEST.json>) contains both literal decision lines, explicitly **NOT received**. A missing-ACK check of the exact frozen command was rejected before worker/model/optimizer construction and created no real claim, ledger, training output or approval. [Preparation receipt](<../artifacts/wave7-preparation/PROPOSAL-001/receipt.json>) records CPU-only initialization (`cuda_initialized=false`) and30.17s preparation, not pilot runtime. The historical Wave5 draft remains unapproved and unchanged.

Exact future guarded command, cwd `F:\Code\HiFT` (shown for review only; missing approval is intentionally fatal):

```text
F:\Code\HiFT\.venv\Scripts\python.exe F:\Code\HiFT\scripts\run_b2_guarded.py --proposal F:\Code\HiFT\artifacts\wave7-preparation\PROPOSAL-001\UNAPPROVED-PROPOSAL.json --authorization F:\Code\HiFT\artifacts\wave7-authorization\W7-COOPERATIVE-PROPOSAL-001.approved.json --output F:\Code\HiFT\artifacts\b2-pilot\B2-W7-COOPERATIVE-PILOT-001
```

**Stopped awaiting both exact human decisions. No authentic training has run.** No real approval file or consumed real grant exists. [Compact validation metadata](<../experiments/wave7_validation.json>) records pilot attempts/updates, DEV losses, gradient traces, training resources and training runtime as null/not measured, not invented zeros.

## Scientific interpretation and next experiment

Authentic execution reliability, gradient/loss traces, actual-run DEV0/25/50/100, fixed-last preliminary learning signal, checkpoint/backbone invariants and cumulative pilot runtime remain **not measured** pending exact approval. Missing measurements are not zeros. Synthetic tests establish engineering behavior only, not B2 trainability, convergence, tracking AUC, UAV generalization, reproduction or R1 novelty.

**Exactly one recommended next experiment:** the one frozen first authentic corrected-B2 grader-only reliability pilot, only after both exact human decisions, with100 effective updates/1600attempts/100windows/900 cumulative seconds, FP32/TF32off, frozen ImageNet backbone and all BN, seeded random grader, unchanged recipe and fixed DEV0/25/50/100, no retry/extension/benchmark panel. Rejecting Policy B leaves strict Policy A blocked unless genuinely sufficient classification evidence or an appropriate exclusive environment becomes available.
