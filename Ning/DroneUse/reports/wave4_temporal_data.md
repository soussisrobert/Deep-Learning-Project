# Wave4 bounded authentic temporal-source pilot

## Scope and authorization boundary

This is **CPU-only original-source preparation**, not real-data SGD, benchmark inference, B0/B1 reevaluation, B2 reproduction, R1 training, biometric identification, or a claim of global physical-person identity. All old frozen code/reports/data were left untouched. Only the new temporal-source module, CLI, tests, this report, and new ignored source/data roots were written.

## Original HiFT VID/YTBB investigation

- Pinned upstream [VID parser](../third_party/HiFT/training_dataset/vid/parse_vid.py) reads original ImageNet VID XML `trackid`, class, occlusion and per-frame boxes under video-specific paths. The [snippet converter](../third_party/HiFT/training_dataset/vid/gen_json.py) splits gaps and drops isolated frames; its generated crops/snippets are not an exhaustive simultaneous-instance identity source or proof of global identity. No raw VID XML/image assets or prescribed prepared VID cache were found in the bounded workspace search. This is not a claim about every disk or unavailable external caches.
- The original [2015 ImageNet VID page](https://www.image-net.org/challenges/LSVRC/2015/) returned HEAD200 and an exact1024-byte206response; the page is **not declared broken**. Author download/login/request terms are distinct from a ready bounded source of original image/XML tracks. No credentials, unapproved image archive, or third-party mirror were used. The author [2017 challenge description](https://image-net.org/challenges/LSVRC/2017/index.php) describes30fully labeled classes per video; no verified canonical cross-video physical identity map was established.
- The pinned upstream [YTBB parser](../third_party/HiFT/training_dataset/yt_bb/gen_json.py) uses video/class/object keys and skips absent/missing images. The author [YTBB download specification](https://research.google.com/youtube-bb/download.html) states **at most one object is tracked per segment**, multiple segments may share a YouTube video, and the segment key is `(youtube_id,class_id,object_id)`, not a global physical-object identity. Normalized coordinates and original millisecond timestamps must be preserved; the legacy seconds-based crop naming is not original media alignment proof.
- Official YTBB detection-train metadata HEAD200, size59,724,774bytes, and a strict1024-byte206probe succeeded. It is **available**, not declared broken; no full57MiBmetadata file or YouTube media was downloaded merely to claim progress. The author warns that media may be removed. No local YTBB CSV/original/prepared frame cache was found in the bounded workspace search. Sparse segment labels do not guarantee an exhaustive simultaneous distinct-instance negative witness, so different segment/video keys must not become assumed known negatives.

Actual probe evidence: [availability ledger](../artifacts/wave4-source/availability_probes.json). Four HEADs and four exact1024-byte ranges consumed4096response-body bytes; no full-body fallback was accepted.

## Why the authentic MOT17 TRAIN alternative

The publisher [MOT17 archive page](https://motchallenge.net/data/MOT17/) confirms all three detector variants are the same source videos, with updated GT relative to MOT16. The service/evaluation front end is archived, but the publisher explicitly says dataset archives remain available. Actual source probe:

- `https://motchallenge.net/data/MOT17.zip`, size **5,860,214,001bytes**;
- Last-Modified `Fri,18Dec2020 17:04:22GMT`, ETag `"15d4bc4f1-5b6c01991f807"`;
- HEAD200, GET206 with exact Content-Range/length, no full large ZIP downloaded;
- official labels-only archive10,107,022bytes also passed the tiny206probe, but was not downloaded redundantly;
- [ZIP64 directory inventory](../artifacts/wave4-source/INDEX-001/inventory.json) contains33960members and required3,917,256new range payload bytes, with [index ledger](../artifacts/wave4-source/INDEX-001/network_ledger.json).

The author [MOTChallenge paper](https://arxiv.org/html/2010.07548v2), AppendixC.2/Table11, defines **nine-column original GT**: frame,id,left,top,width,height,mark,class,visibility. Do not confuse this with the generic ten-column submission format. Coordinates/frame IDs are **1-based**. Canonical source keeps raw boxes and explicitly declares conversion `x-1,y-1,w,h`; widths/heights are not shifted. Time is `(native_frame_id-1)/seqinfo.fps`.

Critically, author annotation rules assign a **new ID after FOV exit/reentry or ambiguous long occlusion** (AppendixC.1.3/C.1.5). An author ID is therefore a **trajectory fragment within one release/video**, not a global physical person. The older author [MOT16 annotation rules](https://arxiv.org/html/1603.00831v2) state the same limitations. No face matching, appearance-based reidentification, or cross-video physical identity inference was performed.

## Preregistered source/split/frame panel

The immutable [selection](../artifacts/wave4-source/SELECTION-001.json), SHA-256 `7a0e43c653d129be13aa7389417895c7b391949472d1387ee8ecbca056ca3971`, was frozen **before images or pair outcomes** after final lead approval. Earlier chat-only suggestions were not acquired/frozen; the selection records the provenance-only reason for the final change and the2x20window rule. No earlier selection artifact was overwritten.

| Canonical source video | Pilot split | Native FPS | Complete source length | Selected inclusive native frame ranges |
|---|---|---:|---:|---|
|MOT17-02-DPM|train|30|600|117–136;407–426|
|MOT17-09-DPM|train|30|525|102–121;354–373|
|MOT17-10-DPM|train|30|654|127–146;444–463|
|MOT17-05-DPM|development|14|837|164–183;572–591|

**160original frames total**,40per video. Each20-frame contiguous window starts at `1+floor((N-20)*q)`, `q∈{0.2,0.7}`. Thus observed within-window positive gaps can be1..19; cross-window gaps exceed the declared positive limit and must not be filled/interpolated. Source frame/time/split choices were not adjusted for pair acceptance, model outputs, loss or accuracy. Both native640x480 and1920x1080source images are retained without reencoding/resizing.

DEV05 is the publisher-identified **ETH/Ess/aess recording collection**, held out conservatively including known05/06detector/MOT16aliases and ETH-Pedcross2/Bahnhof/Sunnyday names. TRAIN02 comes from the Vimeo60139361/MOTChallenge2015 recording;09/10 are conservatively grouped as the ADL recording family, not counted as independent collections. [MOT15 publisher metadata](https://motchallenge.net/data/MOT15/) supports candidate aliases02/Venice-2,09/ADL-Rundle-6,10/ADL-Rundle-8,05/ETH-Pedcross2 by source/length/scene. These are conservative **reservations, not byte-hash or global physical-person equivalences**. Unknown re-edits, pretraining aliases, or people appearing in different recordings remain unproven.

All DPM/FRCNN/SDP variants are reserved as the same video; only canonical DPM originals are acquired. MOT16 aliases are reserved against split leakage, **without equating MOT16 and MOT17 numeric track IDs**, because GT was updated. All four publisher sources are TRAIN; `development` is this pilot's separate held-out source lineage, not the official benchmark test set.

## Canonical data contract agreed with sampling worker

[Acquisition module](../src/hift_research/temporal_source/acquisition.py) emits schema `hift-temporal-source-v1`, explicit frozen/preflight-only status, source/legal/selection/index/evaluation-reservation identities, and videos containing:

- `video_lineage`, author sequence, train/development split, numeric fps/width/height/full length;
- complete unchanged original GT and seqinfo paths/hashes/CRC plus full-source trajectory anomaly/gap audit;
- selected frames with native ID/timestamp, canonical source-image ID, original path/SHA256/ZIPCRC/byte size, pHash64, dimensions and **all selected-frame original annotation rows**, including ignore/nonpedestrian rows;
- raw source boxes/mark/class/visibility/source-row number and source track ID;
- `identity_scope=author_trajectory_fragment_within_source_video`; canonical identity `[MOT17,video_lineage,track_id]` only within this archive release;
- `presence=bool(raw_mark==1)`, meaning the source active/ignore flag, **not** a visibility/class filter or physical absence inference; mark1withvisibility0 stays presence-true, mark0stays presence-false;
- quarantine reasons rather than invented repairs for nonpositive IDs/boxes, invalid flags, duplicate frame/track rows, source-class inconsistency, or extreme one-frame geometric-jump hints. Geometry hints are not biometric identity judgments. Missing full-GT intervals are recorded, never interpolated.

Eligibility is independently frozen/implemented by WorkerB: same-video/source-fragment observed positive frames with nonzero bounded gap; known negatives require **simultaneous same-frame distinct source tracks**, mark1/class1/high visibility, safe nonoverlap, and explicit targetAexclusion from the **final actual augmented search footprint centered onB**, with margin. Merely disjoint source A/Bboxes is insufficient because the511crop spans much more than B. No cross-video negatives, no assumed distinctness from two IDs at different times, and no hidden candidate retries are authorized by this source acquisition.

The original full GT supports missing-mid-frame checks beyond the40selected images. It does not invent absent-track boxes or exhaustively prove visibility. Visibility is an author estimated occlusion/cropping fraction, not a perfect human quality label.

## Validation, provenance and bounded network

- Each original selected ZIP member is checked against the frozen directory, strict HTTPS206range/total/length, stable Last-Modified **and ETag**, ZIPCRC, SHA256, full Pillow JPEG decode, second OpenCV CPU decode, dimensions and exact source frame ordering. No original annotation/image bytes are repaired, interpolated, cropped or reencoded.
- Selected detector-variant frame/GT CRC+size metadata must agree. Only canonical bytes are acquired/hashed; CRC metadata agreement is **not** SHA verification of every unacquired detector variant.
- Every selected original frame is screened against the unchanged15-UAV-video/2937-frame [evaluation reservations](../artifacts/wave3-training/FROZEN-001/reservations.json) by fullSHA and the existing global-frame pHash64 routine. Semantic/transcoded/pretraining alias freedom is **not certified** by those screens. Suspect/duplicate selected content remains quarantined and retained, never secretly replaced.
- Global initial bound:128MiBnew network, including source index, images, annotation metadata and probes; no more than160original selected images. Selected image-member span estimate21,776,259bytes (~20.77MiB); complete GT/seqinfo add only bounded metadata. A1MiBdocumentation/probe allowance was reserved before the source reader. No full ZIP fallback or endless retry loop exists.
- [CPU runtime/source context](../artifacts/wave4-source/RUN-CONTEXT-001.json), SHA-256 `4c64ca13111d8309f82b30272c69246646a677be70b761c39f19679ed2c48a6a`, captures source revision/dirty diff/code hashes, exact CPU software/platform, deterministic selection and no-GPU/no-SGD authority during acquisition. Owned source bytes were unchanged between acquisition launch and capture.

### Completed measured source evidence

Acquisition completed successfully with **160original frames**,120TRAIN plus40development. No source/frame was changed, retried under a replacement selection, or dropped based on pair/model outcomes.

- [Canonical temporal manifest](../artifacts/wave4-source/SOURCE-001/temporal_manifest.json), SHA-256 `2660a96d72ed1645b822120d81835f7a66f93b65da9a4648f6a16fcc04bdba54`.
- [Exact range-response network ledger](../artifacts/wave4-source/SOURCE-001/network_ledger.json), SHA-256 `c956f2d026e78818f95ad4ca7c5a7fc6743289f12d73800e97f1cf2fb15efc90`.
- [Independent CPU raw-source audit](../artifacts/wave4-source/SOURCE-001/independent_audit.json), SHA-256 `80819a3141c3d4e125d3b8a9bdc8c34534299c1aedb78a0234f9437a740abc92`, **passed all160frames**. [Audit code](../src/hift_research/temporal_source/validation.py) independently rereads every full original GT CSV, matches every selected annotation/source-row/bbox/mark/class/visibility/presence, verifies that **no selected-frame source rows were dropped/added/reordered**, rehashes/recomputes CRC for every selected original JPEG and full GT, fully redecodes both ways, recomputes pHash/eval-reservation screens, and checks native timestamps/source split/selected frame ordering.
- Sampling WorkerB independently acknowledged strict schema/native-GT/sourceSHA alignment for all160frames and all complete original GT files. That acknowledgment is source verification, **not** pair-pool/negative-footprint acceptance or training approval; those remain separately reported by WorkerB/lead.

| Source lineage | Complete original GT rows retained | Selected original annotation rows | Selected mark0rows retained | Quarantined selected frames/rows | Full-source track-ambiguity hint records |
|---|---:|---:|---:|---:|---:|
|MOT17-02|30003|2064|800|0/0|0|
|MOT17-09|10411|812|439|0/0|0|
|MOT17-10|17450|1063|333|0/0|0|
|MOT17-05|8013|350|73|0/0|0|

Complete originals contain **65877GT rows**; all **4289selected-frame rows** remain, including1645mark0ignore rows. These counts are **not** a count of high-visibility eligible pedestrians or accepted pairs. No suspected source-track reuse/class/duplicate/geometric-jump hints were found by the declared checks, and no selected-frame exact/perceptual evaluation reservation matches were found. This does not prove author annotation perfection, physical identity uniqueness, or absence of unknown semantic/pretraining aliases.

Original full GT SHA-256 values:

| Source | SHA-256 |
|---|---|
|MOT17-02|`2e3ecb488da8886d3200d402b2b08890c6d2879923839444e9b74fa43a551440`|
|MOT17-09|`592f0d5b519c03b35bb1578c33d726460f63abb91ea0c515f87e8d6d76be001d`|
|MOT17-10|`f8cca05d98fa36d8b6f2b807e03d32407d00d40299d679e4da320e42862677c0`|
|MOT17-05|`ed74c54176b21b38cb690eda79a4755cf16e4c4a9ca2f019b0ba8904636174f5`|

Measured source acquisition response payload is **22,265,130bytes**, index3,917,256bytes, totaling **26,182,386archive-response bytes**. Including4096tiny availability-probe bytes gives26,186,482bytes (~24.97MiB). With the preregistered1MiBdocumentation/probe allowance included instead of double-counting probes, the source receipt accounts **27,230,962bytes (~25.97MiB)**, far below134,217,728bytes(128MiB). These are response-body counts/allowance, not exact wire-level HTTP/TLS overhead measurements. No full archive, labels-only duplicate, original VID archive, full YTBB CSV or unselected JPEG body was consumed as a fallback.

Raw source images/GT, network caches and full receipts are ignored local artifacts. Preserve them and their hashes in the lead's handoff package; the committed report alone does not recreate unavailable local cache bytes. The independent audit was added after acquisition as a separate CPU evidence stage; the earlier acquisition context records the source bytes actually present during acquisition, not an assertion that later audit/test/report files already existed.

## Legal and scientific limits

The publisher licenses data **CC BY-NC-SA3.0** with attribution/noncommercial/share-alike requirements; original third-party source conditions may additionally apply. This report is not legal certification of commercial/redistribution rights. Preserve publisher/source citations and applicable terms with any derivative transfer.

This is a **bounded MOT17 alternative temporal contract**, not original HiFT VID/YTBB recipe reproduction, complete video imagery, production-training readiness, robust tracking accuracy, source-resolution gain, or physical-person-disjoint evaluation. A successful source parser or1024-pair CPU preflight cannot grant real-data SGD permission. No GPU, optimizer step, training, or benchmark rerun is part of this worker's task.

## Reproduction and CPU tests

```bash
.venv/Scripts/python.exe -m pytest tests/test_temporal_source.py tests/test_data_acquisition.py
#Final37passed in1.07s:18new temporal-source +19prior bounded range/ZIP safety fixtures.
.venv/Scripts/python.exe scripts/acquire_temporal_subset.py --selection artifacts/wave4-source/SELECTION-001.json --index-ledger artifacts/wave4-source/INDEX-001/network_ledger.json --reservations artifacts/wave3-training/FROZEN-001/reservations.json --data-root data/wave4-temporal/SOURCE-001 --output artifacts/wave4-source/SOURCE-001
```

Fresh output directories are mandatory. Existing artifacts are immutable provenance, not overwrite targets. Existing index cache bytes are rehashed before reuse. Synthetic tests establish parser/window/identity/budget/coordinate/cache invariants only; they are not fabricated real annotations or tracking scores.
