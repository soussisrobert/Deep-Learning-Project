# Wave2 authoritative data acquisition audit

## Completion: authentic complete pilot data obtained

**Measured acquisition result:** 265 original JPEG frames across the preselected complete official UAV123@10fps sequences `uav6`, `wakeboard7`, and `building5`. Every frame was fully decoded on CPU, all official frame IDs and annotation counts match, and original annotation/image bytes were preserved. **This is not all123 sequences and is not a complete benchmark result.** No training, GPU work, generated annotations, ad-hoc downsampling, mirror substitution, or inference was performed by this worker.

Selection was frozen before tracking outputs in [pilot selection](../artifacts/acquisition/pilot_selection.json) and the successful attempt's timestamped [selection record](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/selection.json): choose tiny complete official sequences across distinct UAV/human-water-sport/building categories using archive sizes and official config only. The lead approved this fixed subset before inference. Later tracking results cannot retroactively justify selection changes. Short/compressed-size sampling is a practical preparation bias, not representative accuracy sampling.

Ready paths:

- [Prepared data root](../data/wave2_uav10_pilot002/prepared/): `data_seq/SEQUENCE/*.jpg`, `anno/SEQUENCE.txt`.
- [Original archive-member tree](../data/wave2_uav10_pilot002/original/): original paths and byte-identical images/annotations/config/README.
- [Evaluation dataset manifest](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/dataset_manifest.json): **evaluation-only GT**, full image SHA-256 identity, `completeness:partial`, `invalid_policy:count_as_failure`, source origin null/unresolved.
- [Acquisition record](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/acquisition.json): selected member URLs/offsets/CRC/SHA, decode results and dimensions, exact expected source configuration.
- [Network ledger](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/network_ledger.json) and [aggregate summary](../artifacts/acquisition/wave2_summary.json).

The dataset-manifest SHA-256 is `ece512a1b174abcc80291d92178f720ed0df7dd042fa1021238a4363164ac03a`. A separate [byte-copy verification](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/byte_copy_validation.json) independently rehashed all265 prepared JPEGs and all3 prepared annotations against the original selected members: every copy is byte-identical. Data and acquisition artifacts are ignored by git; preserve them with immutable experiment artifacts. Source code does not redistribute this benchmark.

## Validation findings

| Complete official sequence | Exact original10fps IDs | Decoded dimensions | GT rows | Absent rows retained | Native first GT xywh |
|---|---|---|---:|---:|---|
| uav6 | 000001–000037 | 720×480 | 37 | 1: frame000024 | 387,192,9,8 |
| wakeboard7 | 000001–000067 | 1280×720 | 67 | 0 | 623,299,11,38 |
| building5 | 000001–000161 | 1280×720 | 161 | 0 | 530,525,22,62 |

The six-digit filenames and inclusive ranges come directly from the original [10fps config](../data/wave2_uav10_pilot002/original/UAV123_10fps/configSeqs.m#L22), [uav6 declaration](../data/wave2_uav10_pilot002/original/UAV123_10fps/configSeqs.m#L106), and [wakeboard7 declaration](../data/wave2_uav10_pilot002/original/UAV123_10fps/configSeqs.m#L115), not directory sorting guesses. These selected segments occupy complete original folders. Every ID is present exactly once; every file is an original JPEG that fully decodes; nonabsent boxes are finite positive-size xywh; initial boxes are valid; no frames or annotation rows were dropped.

**Genuine source absence, not corruption:** the original [README](../data/wave2_uav10_pilot002/original/UAV123_10fps/ReadMe.txt#L5) defines NaN as fully occluded/outside frame. `uav6` frame000024 has a full-NaN annotation. The first acquisition attempt intentionally failed closed on that row under the initial too-strict validator, leaving its [failed-attempt transfer ledger](../artifacts/acquisition/WAVE2-UAV10-PILOT-001/network_ledger.json). The final validator recognizes this documented full-NaN pattern, retains it and its JPEG, and distinguishes it from malformed mixed-NaN/infinite/zero/negative-size rows. **All three sequences use count_as_failure**, retaining all265 denominators. No interpolation, GT repair, synthetic annotation, frame skipping, or performance-driven sequence replacement occurred. Raw GT text is unchanged; JSON null encodes nonfinite values without changing alignment.

Selected annotation SHA-256:

- [uav6 raw GT](../data/wave2_uav10_pilot002/original/UAV123_10fps/anno/UAV123_10fps/uav6.txt): `a7629d09f1b51eec40aa87049f0cf3aa9e088803ef79688a08a57fcdd24e463f`.
- [wakeboard7 raw GT](../data/wave2_uav10_pilot002/original/UAV123_10fps/anno/UAV123_10fps/wakeboard7.txt): `be4a48faa83b2d8a8d814e23dd32b88c89e4ccad6d64a348062e54846457d772`.
- [building5 raw GT](../data/wave2_uav10_pilot002/original/UAV123_10fps/anno/UAV123_10fps/building5.txt): `e8a3969e411422cc837f2005fd25dd21a6019bdcd38224ac2a3c7e49c7d1dcb7`.

## All four authoritative source investigations

### UAV123 and UAV20L: official shared full archive

The official [KAUST resource page](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset) links Google Drive object `0B6sQMCU1i4NbNGxWQzRVak5yLWs` with its published resource key. A bounded confirmed download endpoint returned **HTTP206**, original filename `Dataset_UAV123.zip`, exact total size **14,049,397,769 bytes**, and Last-Modified `Fri, 19 Dec 2025 22:05:18 GMT`. [Probe evidence](../artifacts/acquisition/source_probes.json) and [original directory inventory](../artifacts/acquisition/uav123_content/inventory.json) retain the response metadata/member list.

The directory and original config/README were selectively retrieved, not the14GB archive. The original [full README](../artifacts/acquisition/uav123_content/metadata/UAV123/ReadMe.txt#L1-L16) explicitly explains that UAV123 splits some long videos into shorter underscore-named segments and that long-sequence annotations are in UAV20L. Original [full config](../artifacts/acquisition/uav123_content/metadata/UAV123/configSeqs.m) declares both `seqUAV20L` and `seqUAV123`, sharing source frame directories but using distinct ranges/annotations. Therefore UAV20L is **not a second independent image archive** and UAV variants are not independent evidence.

The full directory inventory's smallest image folder was uav6 (109 original30fps frames, ~4.59MB compressed), but no full-variant pilot images were acquired because the approved separate10fps pilot already meets the minimum authentic-data objective. Full UAV123 and UAV20L evaluation remain unperformed.

### UAV123@10fps: official separate downsampled archive, selected members acquired

Official Drive object `0B6sQMCU1i4NbZmFlQmJBVDlLRDg`, filename `Dataset_UAV123_10fps.zip`, exact size **4,688,528,930 bytes**, Last-Modified `Fri, 19 Dec 2025 23:52:28 GMT`, returned valid HTTP206 Range responses. Both endpoints are publisher-linked, not guessed third-party mirrors. Initial unconfirmed Drive requests returned the public virus-scan confirmation HTML; explicitly confirming the public download yielded bounded ranges. No login credentials or authentication barrier bypass was used.

A ZIP64-aware seekable reader fetched the original directory, then original configuration/README, three raw annotation members and complete contiguous image-member spans. `zipfile` checked original member CRCs; every selected file was independently SHA-256 hashed. Config SHA-256: `c30374367362aef85ecb87ecfd16374fbaed9a23e43094619b9f773f6e8a5b4a`. README SHA-256: `4ab99050fdd05a9070d94877fe70eb9001826630ad5a37ca5b4b5e11e73d4948`. Byte-preserving copies adapt the layout for the existing strict importer; no numeric transformations enter stored GT.

### DTB70: official annotations/config available; image share concretely blocked

The official [DTB70 repository](https://github.com/flyers/drone-tracking) was inspected at `c42e1833acfb858ac8f4ec69fa04ab02ac4c19ad`. Its authoritative tree contains GT/config files but only a README under data, not original benchmark frames. Its linked [Baidu share](https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ) returned an HTTP200 page titled `DTB70.tar.gz` with expired/filtered-share messaging and no usable anonymous binary URL. A reliable image archive size could not be established; **size is unknown, not guessed**. No massive archive, altered-weather dataset, or generated/sample mirror was substituted.

Small official [DTB sequence config](../artifacts/acquisition/official_dtb/experiments/util/configDTBSeqs.m) and [Motor2 GT](../artifacts/acquisition/official_dtb/experiments/anno/Motor2.txt) were downloaded by exact commit and independently Git-blob verified:

- Config: 11,160 bytes; Git blob `5c89bb53d1a313a8cc4048a0c51defd3018066fc`; SHA-256 `133670b1d2d1fbd192e47cdb6e2ff49d7dc33a1894b0dc105c53ed6afe3e3d06`.
- Motor2 GT: 1,154 bytes; Git blob `1d668c17a8b495e98e9ae5a1a5b1316bd7f6d2db`; SHA-256 `56e95da1c6c60171f0263c0ad347bbe7eaa176cd473d71d6a367a913223bdf79`.

These are reference assets only, **not an acquired complete DTB70 sequence**. The short config candidates Motor2/Car2/Wakeboarding2 were considered before official UAV range retrieval succeeded; no model performance was used to choose data.

## Coordinate-origin evidence and primary-evaluation decision

The original10fps config proves IDs/ranges and default initialization placeholders, not source image origin. The publisher-linked [annotation-details PDF](../artifacts/acquisition/annotation_details.bin) is419,429bytes, SHA-256 `a78fd08851e395cbe9c9836a68b498de8b82fac2be00365a89a6688d1400cb5e`. Its extracted [text](../artifacts/acquisition/annotation_details.txt) is a sequence/attribute table and **contains no explicit0-based/1-based author statement**.

The publisher-linked `tracker_benchmark_v1.1.zip` (52,672,929bytes) was independently range-inspected. Archived [MATLAB display code](../artifacts/acquisition/official_toolkit/extracted/tracker_benchmark_v1.1/rstEval/drawResultRect.m#L72-L80) calls `imshow(img)` then `rectangle('Position',r,...)` without origin conversion, and [result display](../artifacts/acquisition/official_toolkit/extracted/tracker_benchmark_v1.1/drawResultBB.m#L80-L93) does likewise. This is consistent with MATLAB one-based pixel-center conventions. It does **not prove how every original box was authored**, so the acquisition manifest honestly records source origin as null/unresolved rather than claiming proof from MATLAB alone.

The lead explicitly decided primary B0/B1 will preserve literal pinned upstream numeric behavior: **initial_bbox = raw firstGT + [0.5,0.5,0,0], no subtraction of1**, and evaluate predictions against raw GT with `annotation_native`. Operational tracker `coordinate_origin:0` is disclosed separately from unresolved source annotation origin. Retrospective physical source-support diagnostics may assume origin1 and subtract1, but must label that as a sensitivity assumption, never change primary boxes/scores, and never feed GT to inference. This avoids silently conflating geometric normalization with legacy reproduction.

## Sequence-level official attribute provenance

Official annotation files are sequence-level12-flag vectors in these original archive members:

- `UAV123_10fps/anno/UAV123_10fps/att/uav6.txt`.
- `UAV123_10fps/anno/UAV123_10fps/att/wakeboard7.txt`.
- `UAV123_10fps/anno/UAV123_10fps/att/building5.txt`.

The original three24-byte comma-separated vectors were subsequently extracted into a **new immutable** [attribute receipt](../artifacts/acquisition/WAVE2-UAV10-ATTRIBUTES-002/attributes.json), with CRC/SHA-256/order/source ranges, without modifying existing acquisition/dataset manifests or frames. Receipt SHA-256: `dc219d04c5c1356183ed0e1d70d079fe4251935cb8556986fcde5b5ec4c35be1`. Active official flags are uav6: `SV,LR,FM,FOC,POC,CM,SOB`; wakeboard7: `SV,ARC,VC`; building5: `CM`. An initial optional parser assumed whitespace and failed on the original comma-separated vector; raw bytes were preserved and a fresh receipt correctly parsed the original comma delimiter. Successful extraction consumed296bytes, with96bytes from that first attempt. They are also machine-discoverable in the preserved [official directory inventory](../artifacts/acquisition/uav10_content/inventory.json); the annotation PDF lists sequence-level attributes and definitions near its table/footer. The original order is `SV, ARC, LR, FM, FOC, POC, OV, BC, IV, VC, CM, SOB`. Relevant provenance categories include full/partial occlusion, low resolution, fast motion, background clutter, and similar objects. **No separate official blur flag was established.** The PDF concerns source sequence metadata; it is not a per-frame10fps event label. These references are provenance only: no invented frame labels, per-frame GT inference feature, automated difficulty reassignment, or performance-stratified reselection was introduced.

## Budget and provenance limitations

Main acquisition/protocol script-consumed HTTP response payload subtotal: **40,674,913 bytes =38.7906MiB**. The optional official attribute receipt adds392bytes including its failed parser attempt, making the final recorded script payload **40,675,305 bytes**, itemized in the separate [budget addendum](../artifacts/acquisition/WAVE2-UAV10-ATTRIBUTES-002/budget_addendum.json). This includes both source directories, metadata, the failed first uav6 validation transfer, the successful18,124,403-byte pilot transfer, original PDF, official toolkit subset, and pinned DTB references. [Aggregate ledger](../artifacts/acquisition/wave2_summary.json) itemizes all components. Small separate web documentation/API reads and HTTP/TLS overhead are excluded from this exact script count; the hundreds-of-MiB remaining margin makes the512MiB initial ceiling unambiguous. There was no large unbounded fallback request.

Full archive SHA-256 values are **unknown** because the huge archives were not fully retrieved. Publisher URL, stable archive size/Last-Modified, retained exact range bytes, ZIP CRC/offsets, and selected-member SHA-256 are the source-version evidence. They establish reproducible selected bytes, not a fabricated complete archive digest or guarantee that the publisher will never replace a Drive object. Reproduction fails closed on changed archive metadata. Third-party sample/webdataset search candidates were not used after the authoritative method worked.

## Implementation and measured CPU tests

Owned implementation: [bounded reader](../src/hift_research/data_acquisition/range_zip.py), [official acquisition/validation adapter](../src/hift_research/data_acquisition/uav.py), [CLI](../scripts/acquire_benchmark.py), [tests](../tests/test_data_acquisition.py), this report and [procedure](../docs/wave2_data.md). Existing evaluator/inference/upstream/root configuration were not modified by this Wave2 worker.

Measured command:

```text
PYTHONPATH=src python -m pytest tests/test_data_acquisition.py tests/test_evaluation.py
Python3.13.0 / NumPy2.4.4 / Pillow12.1.1 / pytest8.4.2 / Windows
54 passed in1.07s  (19 acquisition fixtures +35 evaluator fixtures)

.venv/Scripts/python.exe -m pytest tests/test_data_acquisition.py tests/test_evaluation.py
54 passed in0.91s  (locked project environment; same54 fixtures)
```

Acquisition fixtures cover exact206/range metadata; cache/seek/ZIP access; byte caps before requests; no full-body consumption when Range ignored; size/Last-Modified changes; truncation with received-byte accounting; chunk-boundary reads; unsafe parameters; path traversal; encrypted/symlink/oversized members; no overwrite; parsing MATLAB without execution; complete ID sets and actual JPEG decoding; missing/extra/corrupt images; full-NaN absence retention; malformed boxes; and missing annotation rows. Generated tiny JPEGs exist **only in temporary unit fixtures**, never as acquired benchmark data or annotation substitutes.

Remaining limitations: partial/pilot sampling only; DTB images unavailable through inspected official share; source coordinate origin not explicitly established; full archive hashes and full dataset-level evaluation unavailable; training/pretraining leakage provenance unresolved. Matched real inference, model comparisons, profiling, and retrospective diagnostics are lead-owned separate evidence—not acquisition claims.
