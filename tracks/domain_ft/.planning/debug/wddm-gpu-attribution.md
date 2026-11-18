---
status: blocked
trigger: "User rejects repeated attestation requests and asks to work on the Windows detection blockers."
created: 2026-10-24T13:23:00+08:00
updated: 2026-10-24
---

## Symptoms
- Expected: mechanically distinguish foreign compute from ordinary desktop GPU use on the allocated Windows host, retain native lease/resource/deadline safeguards, no additional attestations.
- Actual: NVML compute and graphics v3 APIs report the same entire WDDM client PID set; all per-process memory unavailable. Current lease parser rejects every C+G entry.
- Reproduction: current System32 SMI XML and direct NVML queries in artifacts/wave6-verification/GOAL-ROUND-004.json. Clean implementation baseline b8269481be8ddec3298540b9ed530f58db87ddc2; original626tests pass.
- User already allocated all compute. Do not ask for that allocation again.
- No real-image backward/SGD, benchmark, recipe/data/checkpoint/upstream/frozen29 edits, process-name allowlists, unrelated process kills or driver configuration changes.

## Current Focus
hypothesis: The concrete native blocker is missing context-type capability, not demonstrated foreign compute. Exact physical adapter reports WDDM 2.7, where process-adapter ClientHint bytes are reserved; engine counters lack a complete API/context inventory.
test: Native PDH/DXGI/D3DKMT and installed TDH schema probes executed; TDD corrected version-aware decoding and unique PCI mapping; 19 artifact-only CPU/injected-native tests passed; standalone capability candidate executed successfully.
expecting: Lead integration reports typed UNSUPPORTED_CLIENT_HINT and UNKNOWN ownership instead of claiming competing compute, while preserving fail-closed resource/lease/launch gates.
next_action: Production integration and full review are complete. Deliver clean repair commit, keep G5/G6 blocked without inventing native ownership evidence, and do not repeat attestations/inventory polls. Further closure needs a genuinely context-complete native primitive/environment, not a reserved-field/idle-counter shortcut.
reasoning_checkpoint: Missing capability is not BUSY. Zero engine utilization/reserved zero/DirectX hint cannot prove absence of idle or mixed compute contexts. Native strong PCI binding does not prove virtual-alias coverage. Project mutex is not a hardware-wide lock.
tdd_checkpoint: Artifact tdd-red001: 2 passed/7 expected missing-helper errors; version-aware decoder and unique binding added; tdd-green001: 9 passed; candidate-tests001: 19 passed. All fixture tests CPU-only, no production src/tests changes by task-21.

## Evidence
- Existing independent review verified NVML v3 ABI24/offset8 and successful calls; overlap is real native output, not XML parser-only error.
- Existing resource headroom passes at last snapshot; snapshot values are not persistent capacity guarantees.
- timestamp: 2026-10-24T13:28:03+08:00
  evidence: First native PDH/DXGI/D3DKMT prototype completed in 0.97 seconds. Unique physical mapping UUID GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f -> NVML PCI 00000000:01:00.0 -> DXGI/D3DKMT LUID 0x00000000_0x00018413. A second identical-description adapter has invalid PCI coordinate; never bind by name. NVML DLL lacks nvmlDeviceGetLuid export.
- timestamp: 2026-10-24T13:30:01+08:00
  evidence: Native KMTQAITYPE_DRIVERVERSION returned 2700 with status0 on exact physical adapter. SDK d3dkmthk.h gates process-adapter ClientHint at >=WDDM2.9; byte264 is Reserve on WDDM2.7. Compiled ABI query size808/result24/querynode800/processadapter272/hint264 validated. Lead canceled unnecessary CUDA calibration of an unsupported field; no GPU control ran.
- timestamp: 2026-10-24T13:32:27+08:00
  evidence: Read-only installed DxgKrnl provider and TdhGetManifestEventInformation schemas succeeded without enabling tracing. Context30/31/32v0 has12 fields but no ClientHint; Device27/28/29v1 has ClientType with no established CUDA classification. No trace/session/privilege request or OS change.
- timestamp: 2026-10-24T13:34:49+08:00
  evidence: Corrected native run003 completed in1.014 seconds; 562 OS PIDs attempted, same before/after PID sets, physical adapter36 successful process-adapter queries/300STATUS_INVALID_PARAMETER/225access-denied/PID0invalid87. All successful rows report adapterNodeCount10, not contextcount. All10 native engine ordinals covered; labels have no separate compute engine. Version-aware ClientHint now null, reservedraw0 retained, no graphics-only inference. Output/source snapshots immutable.
- timestamp: 2026-10-24T13:37:34+08:00
  evidence: Standalone capability_candidate.py executed under10-second external bound with exact UUID readback, unique PCI matching, absolute System32 targetDLL paths, and complete COM/GDI/NVML cleanup. Returns UNSUPPORTED_CLIENT_HINT, competing_compute UNKNOWN, idle_context_absence_proven false; not BUSY. Artifact19tests passed incl injected native failures/cleanup/ambiguousPCI/reserved5/newDX12-notabsence. Candidate SHA383b1cc8322a3457b644ec7ff20374c972bf39e84b342b645a344ed4381d001f.
- Full technical handoff and executable commands: [HANDOFF](<artifacts/wave6-native-debug/pdh/HANDOFF.md>); [candidate](<artifacts/wave6-native-debug/pdh/capability_candidate.py>); [native result](<artifacts/wave6-native-debug/pdh/capability001/capability.json>); [test evidence](<artifacts/wave6-native-debug/pdh/candidate-tests001/stderr.txt>).

## Eliminated
- hypothesis: Another identical SMI/compute-process inventory poll will resolve WDDM attribution.
  evidence: Repeated inventories and direct v3 queries agree without type disambiguation.

- hypothesis: Zero PDH engine utilization or full engine-ordinal coverage proves no idle compute context.
  evidence: PDH exposes engine activity, not complete per-context API identity; all10 engines covered but no separately typed compute engine. Process-adapter NodeCount equals adapter population for every successful PID. Idle/mixed contexts and future creations remain unproved.
- hypothesis: Process-adapter raw0 means a supported ClientHint negative observation on this host.
  evidence: Actual physical WDDM version2700 is belowSDK ClientHint availability2900; field is Reserve. Injected raw5 is also ignored on2700. No GPU workload can make that reserved field supported.
- hypothesis: DXGI adapter descriptions uniquely identify the physical GPU.
  evidence: Two RTX5070Ti descriptions share IDs but different LUIDs; only LUID18413 has the exact UUID-selected PCI coordinate. Virtual alias coverage remains unknown.

## Lead integration and final checks
- Added production capability/activity probes, per-boundary claim/direct-parent lease verification, and strict native inventory/resource validation. Added115tests; full [FULL-002 receipt](<artifacts/wave6-native-debug/FULL-002.xml>) is741PASS/0fail/error/skip in176.26console seconds.
- TDD reproduced38validation failures, including actual false-ready cases; four already returned ineligible but lacked typed malformed-counter rejection. Do not describe all38 as false-eligible cases.
- [Integrity receipt](<artifacts/wave6-native-debug/INTEGRITY-002.json>):217source/test fingerprints (22pre-existing tests newly covered, not newfiles),29frozen sources,1319assets,5named inputs,140prior files,33Wave4 refs,26older refs all unchanged except the intentional two Wave6 source repairs plus six added files. Native leases released.
- [Production diagnostic](<artifacts/wave6-native-debug/INTEGRATED-001/integrated-availability.json>) executed under10seconds inclusive native Job supervision; all3processes empty, no Torch/GPU workload/SGD. Field capability is unsupported on2700; activity API works but blends3D/compute and omits idle contexts.
- Final [TDH map check](<artifacts/wave6-native-debug/pdh/etw-clienttype/RESULT.md>) resolves DeviceClientType natively toLEGACYUSER/USER/CDD, ContextFlags toSystem/Paging/Companion; neither is API-family/compute classification. Native maps plus14offlinechecks pass; no tracing enabled or CUDA control needed.
- [Independent source/evidence review](<reports/wave6_native_review.md>) passes scoped engineering. Remaining G5/G6 block is native context-type capability, not missing user allocation/attestation. No new real proposal, grant, authentic backward/SGD or benchmark.

## Resolution
root_cause: Native telemetry capability gap: NVML C+G membership lacks per-client compute-vs-graphics discrimination; Windows physical interface is WDDM2.7, so process-adapter ClientHint is reserved. PDH/statistics/installed context schemas examined do not supply supported complete idle-safe API/context coverage. These facts do not prove any foreign compute process is competing.
fix: Task-21 delivered isolated standalone typed capability candidate plus corrected native evidence decoder. Minimal valid integration distinguishes UNSUPPORTED_CLIENT_HINT/UNKNOWN from observed BUSY without relaxing launch/resource/lease safety. Lead owns production integration; complete G5 ownership proof remains unresolved.
verification: Native exactUUID->PCI->DXGI/D3DKMT binding and driver2700 status0; no cleanup errors; source snapshots and raw receipts preserved;19 artifactCPU/injected-native tests pass. No Torch/GPU workload/SGD/driverchanges/foreignkills/attestations.
files_changed: Only artifacts/wave6-native-debug/pdh/ and this existing debug-session file. No task-21 edits to src/tests/current reports/frozen29/config/upstream/data.
