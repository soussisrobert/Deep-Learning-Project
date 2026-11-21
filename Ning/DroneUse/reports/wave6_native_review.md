# Wave6 native detection and continuous-authority independent review

## Disposition

**Engineering review PASS for the frozen repairs and inspected evidence; execution remains BLOCKED.** Task-23, reviewer `literature`, 2026-10-24 Asia/Singapore. Both reviewer-raised false-pass families are corrected and reread. No mandatory implementation finding remains open within this review's scope. This is not a real pilot proposal, authorization, GPU allocation, CUDA-control result, or authentic-SGD acceptance.

**G5 remains unresolved ownership/context classification; G6 is dependent-blocked.** Unsupported attribution is not proof of a busy GPU or foreign training. Conversely, mixed/unknown contexts, graphics executable names, zero activity and absent samples cannot establish permission. The latest inspected integrated observation has 34 `C+G` processes, not a graphics-only inventory.

## Scope and provenance

- Reviewed the final [worker](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L24-L86>), [native inventory/lease integration](<F:/Code/HiFT/src/hift_research/wave6/lease.py#L117-L265>), [typed capability probe](<F:/Code/HiFT/src/hift_research/wave6/gpu_attribution.py>), [activity sensor](<F:/Code/HiFT/src/hift_research/wave6/gpu_activity.py>), and all four added test modules. Checked their call chains into unchanged runtime/adapter/entry supervision and local NVIDIA/Windows SDK declarations.
- This reviewer ran only read-only Git identity/diff/status and SHA256 commands. I did not run tests, native API/GPU queries, Torch, models, authentic-image reads, backward, CUDA controls, grants, leases, driver/OS changes, process termination or commits. My only write for task-23 is this report, after Lead's final source freeze. Test/native receipts below were produced by Lead/other workers, independently read and fingerprinted here.
- Base HEAD is `b8269481be8ddec3298540b9ed530f58db87ddc2`, with eight staged implementation/test paths at review time; no pre-commit clean-tree claim is made. [Final integrity receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/INTEGRITY-002.json#L70-L80>) records source/tests/config scoped diff SHA `354bd70fc5815c5519a642e37b0ee8bdbb75ad410e39fc24684b9491ed555ae7` and source-map SHA `4c24203cee880e36941301602c7dd6badbc3b5998ec1c693d27c5a032dfea37b`.
- The map covers **217** sources/tests: compared with the older 189-file inventory, **22 pre-existing tests are newly included in coverage**, not newly authored source. Actual additions are two modules and four test modules; only two old executable files change. All 23 Wave6 module/test/bootstrap raw hashes independently checked below match that receipt.
- Independently rehashed all **29 frozen critical sources** against the [historical raw-byte reference](<F:/Code/HiFT/artifacts/wave5-preparation/PHASE-A-001/wave4-verification.json#L6-L36>): all match. Upstream remains clean at `992aa7c1605360d607af84cb804b93cd8ee81caf`. [Frozen configuration](<F:/Code/HiFT/configs/B2-Pilot-W4.json>) remains SHA `420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e`.
- The inspected integrity receipt additionally reports zero mismatches for **1,319 authentic assets, 5 named inputs, 140 prior files, 33 Wave4 references and 26 older references**, plus the 29 frozen sources; I did not reread authentic assets. It records all native leases released, no real training/grant and no random CUDA control. The [earlier independent review](<F:/Code/HiFT/reports/wave6_independent_review.md>) remains byte-identical, SHA `f41911fae4d605e03131b9dfb8ca4e206e8508831c9360157d6c9863d7b1e064`.

## Concrete findings and closure

### F1 — duplicate critical XML nodes could hide competing/unknown inventory — FIXED

The earlier parser used first-node selection. An empty first process group followed by a second populated `C+G` group, or a process containing first `type=G` then `type=C+G`, could incorrectly return eligible. Duplicate identity/memory fields and nested scalar data were likewise ambiguous. This was a genuine malformed-telemetry false pass, not evidence of an actual graphics-only GPU.

[Duplicate RED receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/DUPLICATE-RED-001.xml>) records **10/10 reproduced failures**. Final [single-node and leaf checks](<F:/Code/HiFT/src/hift_research/wave6/lease.py#L117-L158>) require exactly one UUID, memory group/total/used/free, process group, PID and type; scalar children are forbidden. The [regressions](<F:/Code/HiFT/tests/test_wave6_inventory_integrity.py#L29-L49>) are included in the final passing suite.

### F2 — ignored mixed container text could still turn unknown into empty/graphics-only — FIXED

The first repair still ignored non-whitespace text before a valid PID/type or after a process group. The follow-up also covers root, GPU and memory-container text. [Mixed-content RED receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/MIXED-CONTENT-RED-001.xml>) records **5/5 reproduced failures**. Final [container validation](<F:/Code/HiFT/src/hift_research/wave6/lease.py#L124-L177>) checks root, selected GPU, memory group, process group and rows; the sibling-tail checks close the reported cases. Inventory coverage is now **43 cases** in the final passing suite.

Lead's earlier [inventory RED receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/INVENTORY-RED-001.xml>) separately establishes 23 failures / 5 passes before the initial repair. Those unavailable-inventory, invalid/duplicate PID, nonfinite/bool/string/negative host-counter and inconsistent-resource cases remain covered. Across the three preserved RED runs, these are **38 reproduced validation failures, including false-ready cases**, not 38 proven false-eligible results: four initial malformed host cases already returned ineligible but lacked the required typed malformed-input rejection. These finite regression families are not a claim that every conceivable driver/XML corruption has been exhaustively proven impossible.

## Call-chain and native ABI assessment

### Continuous authority and unchanged training/budget semantics

[Live authority](<F:/Code/HiFT/src/hift_research/wave6/worker.py#L64-L86>) is no longer only a startup check. Every runtime source boundary rechecks the one-use claim against the actual direct-parent PID and the live physical-UUID lease/digest/nonce before source/data rehashing. Underlying creation-time identity, lease-byte and native mutex checks remain in force. Synthetic CPU execution still rechecks its claim but does not verify/acquire an authentic GPU lease; dispatch mode is strictly a matching boolean before Torch.

[Runtime source checking](<F:/Code/HiFT/src/hift_research/wave6/runtime.py#L154-L157>) reaches this hook at the existing [post-backward, pre-clip, pre-step and post-step guards](<F:/Code/HiFT/src/hift_research/wave6/adapter.py#L39-L59>), with initial/evaluation/checkpoint/final checks retained. The 15 [live-authority tests](<F:/Code/HiFT/tests/test_wave6_live_authority.py>) inject loss/reuse/drift and verify ordering and fresh repeated checks. These are CPU/metadata tests, not a real-SGD demonstration.

The additions do not change loss, optimizer, RNG recipe, source/data policy, batch/window/update/attempt limits, initialization digest or checkpoint/recovery rules. Native observations consume no Torch RNG. [BOOT origin and deadline](<F:/Code/HiFT/scripts/run_b2_guarded.py#L7-L9>) and [entry parent guard](<F:/Code/HiFT/src/hift_research/wave6/entry.py#L58-L98>) are byte-unchanged: availability, capability/activity calls, lease acquisition and cleanup remain inside the inclusive external budget. Native functions are not individually interruptible Python calls; the existing external parent/Job containment is essential, not an optional per-call timeout claim.

### Exact physical capability, not a context-absence classifier

[Capability implementation](<F:/Code/HiFT/src/hift_research/wave6/gpu_attribution.py#L40-L163>) asserts Windows/x64 pointer/LUID/DXGI/PCI/open/query sizes `8/8/312/68/12/24`, with query-data offset 8. COM enumeration/GetDesc1 indices 12/10, IID, U32 handle/PCI fields and KMT query identifiers 6/13/30 are consistent with installed headers and the inspected [ABI evidence](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/HANDOFF.md#L75-L95>). Only KnownDLL kernel32 bootstraps; NVML/DXGI/GDI load from native System32 paths.

The probe verifies supplied UUID by readback, maps NVML PCI domain/bus/device/function plus vendor/device/subsystem to a unique DXGI LUID, rejects nonzero-domain/linked/ambiguous mappings, and makes no virtual-alias coverage claim. The inspected native mapping is PCI `00000000:01:00.0` to LUID `0x00000000_0x00018413`. The other identical-description adapter has a different LUID and sentinel PCI coordinates; names do not merge it into, or exclude it from, consumer coverage.

Installed [SDK conditional](<C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared/d3dkmthk.h#L3710-L3715>) makes the process-adapter field `ClientHint` only at WDDM >=2.9; below that it is `Reserve`. Native **2700** therefore correctly yields `UNSUPPORTED_CLIENT_HINT`, not busy, CUDA-negative, or graphics-only. Newer layout availability is only `SUPPORTED_UNCALIBRATED`. Cleanup uses finally blocks for NVML, KMT and COM; failure changes the observation to native-error/unknown. The 20 [capability/integration tests](<F:/Code/HiFT/tests/test_wave6_gpu_attribution.py>) cover these distinctions without native workloads.

### Bounded activity telemetry, never authorization

[Activity layouts](<F:/Code/HiFT/src/hift_research/wave6/gpu_activity.py#L22-L31>) match the [MSVC-compiled NVIDIA header evidence](<F:/Code/HiFT/artifacts/wave6-native-debug/nvml/BUILD-003/abi_layout.json>): utilization-v1 sample 40 bytes, descriptor 24, legacy sample 32, relevant pointer/timestamp offsets 16/8, descriptor version `16777240`. Both documented utilization APIs have correct argument/return bindings; Windows/x64 uses the unified calling ABI.

The sensor bounds allocation at 512 records and native calls at eight including cleanup. It allows legacy fallback only for explicit unsupported/function/version errors or an absent export, not permission failure. It checks returned counts, caller-owned pointers, buffer lengths, percentage/sentinel fields, PID/timestamp validity, future/rollback conditions, duplicate conflicts, overlap, age and cursor ordering. Missing legacy fields/native-unavailable values stay null. UUID readback, hash-stable System32 DLL identity and finally shutdown are enforced. The 37 [portable CPU tests](<F:/Code/HiFT/tests/test_wave6_gpu_activity.py>) explicitly forbid unmocked native loading and label replay values synthetic.

Installed [NVML declarations](<C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include/nvml.h#L1513-L1536>) explicitly define SM utilization as **3D/Compute**. A positive sample is activity, not compute-kind attribution; zero/stale/no samples are not idle-context absence. PID/timestamp rows are not a creation-time-complete ownership inventory. [Preflight integration](<F:/Code/HiFT/src/hift_research/wave6/lease.py#L249-L263>) only appends observations on opaque WDDM inventory, never changes eligibility, and preserves unavailable/error results with false authorization/classification/idle-proof flags. No driver/accounting setters, process kills, workloads or new approval paths are added.

## Final evidence inspected

1. [FULL-002 acceptance receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/FULL-002.xml>): **741 tests, zero failures/errors/skips, 176.247 seconds JUnit** (Lead CLI: 176.26 seconds). This is the prior 626 plus 115 additional cases: authority 15, inventory 43, capability 20, activity 37. I inspected the XML header and independently hashed the complete file. Earlier [FULL-001](<F:/Code/HiFT/artifacts/wave6-native-debug/FULL-001.xml>) retains its three failures and is **not acceptance evidence**. Lead diagnosed spawn re-entry from the ignored harness; the [corrected runner](<F:/Code/HiFT/artifacts/wave6-native-debug/run_full_suite.py#L4-L13>) has a main guard and a never-overwrite result path. No source/limit relaxation is represented by that harness repair.
2. [Production-function native observation](<F:/Code/HiFT/artifacts/wave6-native-debug/INTEGRATED-001/integrated-availability.json>): `eligible=false`, exact GPU UUID, 34 `C+G` rows, WDDM 2700 unsupported ClientHint, competing-compute UNKNOWN, no idle/virtual-alias proof. Host available **37,398,278,144 bytes**, GPU free **14,282,653,696 / 17,094,934,528 bytes**; memory/headroom passes this instantaneous snapshot and is not its blocker. Activity-v1 returns count-query 7, data 0, one fresh SM=4% row and one stale discarded row; init/UUID/shutdown succeed and DLL hash is stable. This does not establish whether that activity is graphics or compute. Receipt states no Torch, no created GPU workload, zero real SGD, diagnostic-only.
3. The [integration worker](<F:/Code/HiFT/artifacts/wave6-native-debug/integrated_probe_worker.py>) calls the production function directly, without mocks. The [integration runner](<F:/Code/HiFT/artifacts/wave6-native-debug/run_integrated_probe.py>) uses the existing parent guard and atomic suspended native Job under a **10-second diagnostic-only** budget, not a pilot grant. [Diagnostic result](<F:/Code/HiFT/artifacts/wave6-native-debug/INTEGRATED-001/diagnostic-result.json#L15-L58>) records exit 0, three total processes, zero active, closed Job, verified tree empty, no forced termination/errors, and 0.5226-second timing. The [authoritative final ledger](<F:/Code/HiFT/artifacts/wave6-native-debug/INTEGRATED-001.time.jsonl>) reaches entry-finished after supervisor cleanup at approximately 0.5243 seconds. Diagnostic COMPLETED means observation/containment completed, **not G5 passed**; no training telemetry/checkpoint is expected or claimed.

The existing [native investigation handoff](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/HANDOFF.md>) and [NVML alternative evidence](<F:/Code/HiFT/artifacts/wave6-native-debug/nvml/LIVE-001.json>) correctly distinguish reserved/unsupported fields, PDH engine coverage, missing complete context attribution and mixed SM activity. The prepared CUDA positive control was not run. This review adds no requirement for another human attestation and recommends no permission workaround, process-name waiver, unrelated-work termination, driver/OS change or real launch.

## READONLY supplement — installed ETW maps resolved

Independently inspected the [result](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/etw-clienttype/RESULT.md>), [native maps](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/etw-clienttype/native001/maps.json>), [manifest](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/etw-clienttype/native001/manifest.json>) and offline verifier/receipt, rehashed the saved raw buffers and checked their enum headers: Device events 27/28/29 define only `DeviceClientType=0 LEGACYUSER / 1 USER / 2 CDD`, while Context events 30/31/32 define flags `1 System / 2 Paging / 4 Companion`, not compute/API kind. All three device-map lookups record native `122→0`, and the [14-check offline receipt](<F:/Code/HiFT/artifacts/wave6-native-debug/pdh/etw-clienttype/verify001.json>) corroborates the raw maps; this concretely eliminates these fields as the missing discriminator rather than leaving their mapping unestablished. No trace session/provider enabling, privilege change or GPU control occurred, no particular CUDA-client payload value or complete live-context inventory was established, and the G5/G6 disposition, 217-source freeze and 741-test acceptance remain unchanged.

## Independently checked fingerprints

Complete reviewed Wave6 source/test/bootstrap SHA256 values; these match the expanded integrity map:

```text
97fbb60ed768167352ca633f724c9f10ff7a76dab426b97c8944fae06be06232  src/hift_research/wave6/__init__.py
2a9231f06f91489e9e91238f3cf224638c626747b8413a9702285668a6b49504  src/hift_research/wave6/adapter.py
6ff8f7be6b448c6b18dbf0ed433939c3039316f125b5a496ac6e4a762595d248  src/hift_research/wave6/authorization.py
037d04f20becd04595fbe2a6ea398c7a922e17a25f7b972220b4fdd2ea8bfcb0  src/hift_research/wave6/entry.py
fd58107dabe71042be4462d8f0fb28060ef4fe322b2244d5e5b65136ab38751a  src/hift_research/wave6/gpu_activity.py
365722f274baed7a04db22bace1a7054f9ac8bb18b1766a652207140c74daef3  src/hift_research/wave6/gpu_attribution.py
9ae22afc3b12dd4a62c98fbbb615170f5c953a7011cb25f22a3a90edf740f577  src/hift_research/wave6/lease.py
49ef6a90757aa0eb2d2c2b20eda0c17c0774878dc2bafbebb4cd471e05d17560  src/hift_research/wave6/ledger.py
51ac230e25d4927d88464f1dcaa2e15e70680fbb13602dc9db281e40d88ff3aa  src/hift_research/wave6/runtime.py
ccf45f7f9e09bfd6553b635c76b2bd0368ac4465ae1a24b139769600a4751631  src/hift_research/wave6/supervisor.py
72a066357a226aa8a4a82f76fb21efcd2bd67a91379909144c6542068d1975d3  src/hift_research/wave6/telemetry.py
85cffb16bbd229ea26668f555fd41a57cbe13dcc64aeb7687aeafef0df77ab7d  src/hift_research/wave6/windows_job.py
0ac5265bb0b3bfef1a00d40903bb153c79d72e95d7843db606221b780db6665f  src/hift_research/wave6/worker.py
fd4f73985fc0e4312b45b7919bcdb79ffba19b5bff358141f26e6adb73e8ac86  scripts/run_b2_guarded.py
7cea04d15e159cec4c5cab1e28e9b8431c897325e4a57d50de87454c95bf1a3c  tests/test_wave6_authorization.py
671d7ef47c435772394e8e854cd1553f80ab40c00673bb9f10479bad07d6fcb2  tests/test_wave6_gpu_activity.py
8a097f55590b36c97c938094f9178f58ab5c823dcae95353cec7e15bec7d3ba8  tests/test_wave6_gpu_attribution.py
0578a23ff02ac6af44c46a3236606d0977c31375f99b4365070e8b33fc79baa6  tests/test_wave6_integration.py
dc99d698a287a67b93c25c65ce202121f328f3ed055ad723435bbd6eb637ad73  tests/test_wave6_inventory_integrity.py
81d02d2b42c1599173ba140dc402c41e8278cbde968eed8e05ea8ce87dfb3bfd  tests/test_wave6_lease.py
7d1b6eb6f3ec75103b520057f4a4c120c507f3ad63f71a776434c1af7660191f  tests/test_wave6_live_authority.py
2fce7194717a9635df002bae0f86247584b5179e54185b83c5e7fb16fe500bdf  tests/test_wave6_runtime.py
230412d77bbee73be1457a831e1d528adb2de0a58cb5cb22b70ffb7a04a96a77  tests/test_wave6_supervision.py
```

Independently hashed final receipts and ignored, spawn-safe test harness:

```text
946af2109b6d1e727186ef1bca05c47c6ae64bf49f466ba30513e9cc917ed135  artifacts/wave6-native-debug/FULL-002.xml
9236c2583eb833f6534290a987463f732a970e68521b96b6f6f4cdb5f0b16485  artifacts/wave6-native-debug/INTEGRITY-002.json
fbfa482632726434bbef68b909cfc8f113e91b5195962fa249f3c29cc1f74c33  artifacts/wave6-native-debug/INTEGRATED-001/integrated-availability.json
fb58493ff0a78284d918edede9742b97b96d3faec61aa74c2dcfd7b9a5d65f28  artifacts/wave6-native-debug/INTEGRATED-001/diagnostic-result.json
21dda8e22863fc9619557a566bef9ffa736ceab887cc1daf55161ab48f3b4388  artifacts/wave6-native-debug/INTEGRATED-001/supervisor_receipt.json
fd0382d0f8bc4b09f891aa7435574be143a323c65ccacdb41bb5708ea2d6d4b1  artifacts/wave6-native-debug/INTEGRATED-001/native-smi.xml
61b1319bbe19ffd81f267993246ff61cae110e6c1272ff3c5ed42fbebf36f071  artifacts/wave6-native-debug/INTEGRATED-001.time.jsonl
7144955def0445f424ef4b69d4f1080b7044176539b5082d58101e6b712407ad  artifacts/wave6-native-debug/run_full_suite.py
```
