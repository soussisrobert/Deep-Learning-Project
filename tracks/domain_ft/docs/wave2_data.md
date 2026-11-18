# Wave2: bounded official data acquisition

## Available now

The project now contains **three authentic complete UAV123@10fps sequences**, not a complete benchmark:

- `uav6`: 37 frames, original IDs `000001`–`000037`, 720×480.
- `wakeboard7`: 67 frames, original IDs `000001`–`000067`, 1280×720.
- `building5`: 161 frames, original IDs `000001`–`000161`, 1280×720.

Every one of the **265 original JPEGs** was fully decoded on CPU. Exact frame IDs, counts, segment boundaries, filenames, and GT row counts match the **configuration shipped inside the official archive**. All image/annotation bytes remain unchanged. Sequence choice was frozen before any model outputs: minimum compressed-byte complete sequences across UAV, human water sport, and building categories. This is a short practical pilot, not representative benchmark sampling.

Use the immutable [dataset manifest](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/dataset_manifest.json) for **offline evaluation only**. Prepared [data root](../data/wave2_uav10_pilot002/prepared/) contains `data_seq/SEQUENCE/*.jpg` and `anno/SEQUENCE.txt`. Original archived paths/bytes are retained under the separate [original tree](../data/wave2_uav10_pilot002/original/). Complete acquisition provenance, SHA-256 hashes, ZIP offsets/CRC, and decode findings are in the [acquisition record](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/acquisition.json), [transfer ledger](../artifacts/acquisition/WAVE2-UAV10-PILOT-002/network_ledger.json), and [summary](../artifacts/acquisition/wave2_summary.json).

These data/provenance paths are ignored by git; preserve them with experiment artifacts, not source commits. Original dataset licensing/citation applies; do not redistribute without checking the publisher's terms.

## Reproduce without downloading entire archives

Requirements: Python, NumPy, Pillow. The lead pins Pillow 12.1.1 in the environment. From the project root:

```bash
python scripts/acquire_benchmark.py --output-root data/wave2_uav10_fresh --provenance-dir artifacts/acquisition/WAVE2-UAV10-FRESH --budget-mib 480
```

Both output directories must be new. The selection is fixed in [acquisition code](../src/hift_research/data_acquisition/uav.py); there is no result-driven automatic sequence replacement. The public [official resource page](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset) links the exact Google Drive object used. The script reads the ZIP64 directory with validated HTTP Range requests and fetches only the original config/README, selected annotations, and contiguous selected image-member spans. It never consumes a full response when the server ignores Range. Every extracted member is CRC-checked and SHA-256 hashed; all image bytes are fully decoded; every row is retained.

The full original 10fps archive is **4,688,528,930 bytes**. The successful three-sequence transfer consumed **18,124,403 bytes**, reusing a previously archived directory cache. A fresh command without cache additionally reads approximately 5.19MB of directory metadata. Across all Wave2 acquisition/probe attempts, directories for both UAV archives, the official annotation PDF, official MATLAB-toolkit snippets, and pinned DTB reference files, script-consumed HTTP payload was **40,674,913 bytes (38.79MiB)**; the later optional attribute receipt adds392bytes, for a final **40,675,305 bytes**, far below the initial 512MiB budget. Small web documentation/API reads and HTTP/TLS headers are outside that exact script ledger, but cannot approach the remaining hundreds of MiB.

The exact archive size and Last-Modified values observed are pinned for fail-closed retrieval. If the publisher replaces an archive, stop and inspect new provenance; do not silently accept a new version. Full archive SHA-256 is **not known** because huge archives were not downloaded. The evidence is official HTTPS URL, stable byte length/Last-Modified, retained range bytes, original ZIP directory metadata/CRC, and selected-member content hashes—not a fabricated full-archive hash.

A prior official inventory cache can optionally be supplied with `--inventory`; its source URL/size and cached-range SHA-256 hashes are validated. Do not use arbitrary third-party inventories. The CLI maximum 480MiB leaves room for prior metadata within the initial 512MiB budget; cumulative use must still be accounted for rather than repeatedly resetting a run budget.

## Genuine absent annotations: preserve them

The official [10fps README](../data/wave2_uav10_pilot002/original/UAV123_10fps/ReadMe.txt#L5) states NaN means the object is fully occluded or outside the frame. `uav6` has one full-NaN GT row at **frame `000024`**, zero-based row 23. The other two selected sequences have no absent rows. All sequences have valid first-frame initialization.

The initial overly strict acquisition attempt failed on this genuine source row. It was not dropped, interpolated, repaired, or used to replace the preselected sequence. The final validator distinguishes explicitly documented **full-NaN absent rows** from malformed mixed-NaN/nonfinite/zero/negative-dimension boxes, which fail validation. All three pilot sequences use **`count_as_failure`**, preserving every expected frame in denominators. Raw annotation text retains NaN; the JSON evaluation manifest uses null values for these entries without changing row alignment.

Evaluation therefore requires the same explicit policy:

```bash
python scripts/evaluate.py --dataset-manifest artifacts/acquisition/WAVE2-UAV10-PILOT-002/dataset_manifest.json --predictions results/EXPLICIT_MATCHED_RUN --invalid-policy count_as_failure --prediction-coordinate-origin annotation_native --output experiments/runs/EXPLICIT_NEW_EVALUATION.json
```

The prediction directory must contain exactly the three corresponding `.txt` or `.csv` files, with one row per frame including initialization. A one-file run needs `--sequence`; it remains partial. This invalid-GT policy is disclosed and is not claimed to reproduce the legacy toolkit's invalid-center precision bug.

## Coordinate origin and exact B0 initialization

**Explicit publisher statement of annotation origin 0 versus 1 was not found.** The original config documents frame IDs, not image coordinate origin. The official annotation-details PDF is a sequence/attribute table; it does not settle coordinate origin. The publisher-linked MATLAB toolkit displays result rectangles unshifted with `imshow` and `rectangle('Position',r,...)`, consistent with MATLAB's usual one-based pixel-center display. This supports an **assumption**, not a conclusive author-origin guarantee. Dataset metadata remains `annotation_coordinate_origin:null`.

For the primary matched B0/B1 pilot, the lead explicitly chose **literal upstream numeric passthrough**, independent of that unresolved origin:

1. Preserve raw GT numeric coordinates for offline evaluation.
2. Compute tracker-ready initialization as raw GT plus `[0.5,0.5,0,0]`, exactly as the pinned upstream rectangle conversion does.
3. **Do not subtract one** in primary initialization or evaluation.
4. Label runner `coordinate_origin:0` as the operational tracker indexing convention, not a claim that the source annotations were authored zero-based. Include `annotation_to_tracker_transform:'upstream_numeric_passthrough_plus_half_pixel_no_origin_shift'` and unresolved source-origin provenance.

Tracker-ready initial rectangles are consequently `uav6:[387.5,192.5,9,8]`, `wakeboard7:[623.5,299.5,11,38]`, and `building5:[530.5,525.5,22,62]`. These are initialization transforms, not generated later-frame annotations. The generic previous asset importer with `--coordinate-origin 1` intentionally performs a one-pixel origin conversion; **do not use that conversion for literal upstream numeric B0 parity**.

A retrospective physical source-support diagnostic may separately evaluate an assumed one-based source origin by subtracting one from GT x/y. That is an offline sensitivity assumption, not primary evaluation, an established publisher fact, or an inference feature. Keep assumption/diagnostic identities separate from native-numeric primary scores.

## All four requested sources investigated

- **UAV123:** official `Dataset_UAV123.zip`, **14,049,397,769 bytes**; true HTTP206 Range support verified. Original directory plus config/README were inspected; images not acquired because the approved 10fps pilot already meets the minimum preparation objective. Shared source folders require exact segment ranges.
- **UAV20L:** distributed in that same official full archive with separate `anno/UAV20L` annotations and `seqUAV20L` config. The original [full README](../artifacts/acquisition/uav123_content/metadata/UAV123/ReadMe.txt#L1-L16) explicitly explains shared videos and shorter UAV123 segments. It is not independent evidence and needs no second giant archive. Images/GT for a complete UAV20L evaluation were not acquired.
- **UAV123@10fps:** separate official archive and matching original config/annotations; the three selected complete segments are acquired. No ad-hoc downsampling or reconstructed annotations were used.
- **DTB70:** official [repository](https://github.com/flyers/drone-tracking) at `c42e1833acfb858ac8f4ec69fa04ab02ac4c19ad` has annotation/config references but no benchmark images. Its official [Baidu link](https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ) returned an expired/filtered share page titled `DTB70.tar.gz`; no usable anonymous image download or trustworthy size was exposed. Pinned config and `Motor2` GT were independently downloaded and Git-blob verified for preparation, **not treated as a complete image sequence**. No generated/altered-weather/sample mirror was substituted.

Official sequence-level12-flag vectors are separately preserved in a new immutable [attribute receipt](../artifacts/acquisition/WAVE2-UAV10-ATTRIBUTES-002/attributes.json), with CRC/SHA-256/source/order. Active flags: uav6 `SV,LR,FM,FOC,POC,CM,SOB`; wakeboard7 `SV,ARC,VC`; building5 `CM`. These are sequence-level metadata only, not per-frame events or inference features; no separate official blur flag was established. Existing acquisition/dataset manifests and frames were not changed.

See the [Wave2 audit](../reports/wave2_data.md) for exact hashes, test evidence, and remaining limitations.
