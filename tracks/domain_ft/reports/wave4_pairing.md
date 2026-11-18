# Wave4 temporal/known-distinct pairing: bounded authentic input evidence

## Decision and measured result

**Reduced temporal-pilot data contract PASS, not full training/reproduction readiness.** Authentic MOT17 TRAIN trajectory-fragment positives and simultaneous source-witnessed negatives now reach checked B2 model inputs. TRAIN1024 fixed attempts produced **973 genuine nonzero-gap positives +50 safe negatives accepted, one negative rejected**, with **zero replacement/retry**. Development is a separately preregistered **128-attempt balanced, unaugmented control**:64positive/64negative accepted. This worker used **no model forward, SGD, or GPU**. G2/G3 claims concerning optimizer/resume/real-forward feasibility belong to the separately owned lifecycle/lead proofs; this report does not promote input correctness into convergence or training authority.

| Evidence | TRAIN | DEVELOPMENT |
|---|---:|---:|
| Source videos / selected frames | 02,09,10 /120 | 05 /40 |
| Frozen eligible ordered positive pairs | 18994 | 1708 |
| Frozen nominal-guarded simultaneous negative pairs | 5192 | 50 |
| Attempts | 1024 | 128 |
| Accepted positive / negative | **973 /50** | **64 /64** |
| Rejected | **1**, actual search sees templateA under guard | 0 |
| Replacements | **0** | **0** |
| Actual endpoint frames represented | 120/120 | 38/40 |
| Source fragments represented | 53/58 eligible-endpoint fragments | 11/11 |
| Fresh replay / persisted archives independently verified | **1024 /1023** | **128 /128** |

TRAIN candidate negatives:51/1024=**4.9805%**, Wilson95%CI[3.8082%,6.4892%]. Accepted-negative fraction:50/1023=**4.8876%**, descriptive Wilson95%CI[3.7268%,6.3859%]. Acceptance1023/1024=99.9023%, Wilson95%CI[99.4489%,99.9828%]. These describe the frozen conditional-pilot sampling process, not population prevalence or independently sampled people. DEVELOPMENT's64/64 class schedule is fixed, **not Bernoulli**; no probabilistic negative-prevalence interpretation applies.

**Important repetition disclosure:** TRAIN973 positive attempts contain957 unique source endpoint pairs (16 repeats, maximum multiplicity3), each with separately seeded augmentation. All50 accepted negative source pairs are unique. DEVELOPMENT64 negatives contain **38 unique source pairs/exact input-array sets**,26 repeats, maximum multiplicity4, because the predeclared control draws from50eligible pairs with replacement. Its64positives are unique. We did **not** retune seeds, extend the cohort, replace pairs, or force uniqueness after seeing this. Balanced128 is not128 independent examples.

## Immutable artifact map

- [Source selection](<F:/Code/HiFT/artifacts/wave4-source/SELECTION-001.json>): SHA256 `7a0e43c653d129be13aa7389417895c7b391949472d1387ee8ecbca056ca3971`.
- [Authentic source manifest](<F:/Code/HiFT/artifacts/wave4-source/SOURCE-001/temporal_manifest.json>): `2660a96d72ed1645b822120d81835f7a66f93b65da9a4648f6a16fcc04bdba54`.
- [Pre-outcome eligibility preregistration](<F:/Code/HiFT/artifacts/wave4-preflight/ELIGIBILITY-PREREG-001.json>): bound selection, seed, gap, negative guard and balancedDEV policy before source pair outcomes.
- [Final TRAIN policy](<F:/Code/HiFT/artifacts/wave4-preflight/FROZEN-002/train_policy.json>): `c0d811b945f385084daa8ed27042e2bc0178a4859ffab34868fda8304d33afa9`.
- [Final DEVELOPMENT policy](<F:/Code/HiFT/artifacts/wave4-preflight/FROZEN-002/development_policy.json>): `15bf57bc1b7107cf5d6ca51ed72a54c520e13fda7080a2014c833bb2afcd6a36`.
- [TRAIN summary](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-002/train/summary.json>) / [every TRAIN attempt](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-002/train/candidate_ledger.jsonl>): ledgerSHA `bfa185342524cc8b2cf44f7551f92c93e77c1a8f9ff946d14cc6d32077ecfaa3`.
- [DEVELOPMENT summary](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-002/development/summary.json>) / [fixed control cohort](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-002/development/candidate_ledger.jsonl>): ledgerSHA `eae252a734198935c62a80fdbd8239d05169c710a7182d6188de2d8d21c1efeb`.
- Independent [TRAIN verification/distributions](<F:/Code/HiFT/artifacts/wave4-preflight/VERIFICATION-002/train_verification.json>), [DEVELOPMENT verification/distributions](<F:/Code/HiFT/artifacts/wave4-preflight/VERIFICATION-002/development_verification.json>), and [repeat/mask/augmentation/containment supplement](<F:/Code/HiFT/artifacts/wave4-preflight/VERIFICATION-002/distributions_supplement.json>).

## 1. Source, native identities and leakage boundaries

WorkerA acquired160official MOT17 TRAIN JPEGs and complete original GT/sequence metadata:40frames/video, two contiguous20-frame windows at20%/70% of each source timeline. Canonical TRAIN lineages are **MOT17-02, MOT17-09, MOT17-10**; **MOT17-05/ETH** is development. Exact windows were frozen before images/pair outcomes:02[117–136,407–426],09[102–121,354–373],10[127–146,444–463],05[164–183,572–591]. TRAINfps30; DEVfps14. Known detector variants DPM/FRCNN/SDP are the **same video**, not independent clips; DPM supplies canonical image files. MOT16/17 lineage aliases and ETH/ADL/Venice alias candidates are reserved conservatively; numeric track-ID mappings across releases are **not** assumed. The source-worker receipts own HTTP206/CRC/budget/legal provenance; this worker did not acquire extra assets.

The independent [parser](<F:/Code/HiFT/src/hift_research/temporal_contract/population.py>) authenticated every selected annotation against complete GT:02=30003rows,09=10411,10=17450,05=8013. Strict native **9-column** format is frame,id,left,top,width,height,mark,class,visibility, with integer frame/track IDs, mark0/1, recognized native classes, finite geometry and visibility[0,1]. Raw one-based x/y are converted exactly to `(x-1,y-1,w,h)`; edge-clipped negative x/y are **not** forbidden. Manifest frame/track/source-row/identity, boxes, flags and visibility must agree with GT. Presence is exactly `bool(mark==1)`, not a substitute for visibility/class eligibility. Endpoints require positive author IDs, mark1,class1,visibility≥0.5 and positive available original-image support.

Source identity is **(MOT17release, canonical video, author trajectory-fragment ID)**, never a global physical-person UUID. Primary [MOT17 paper AppendixC.1.3/C.1.5](<https://arxiv.org/html/2010.07548v2>) explains reentry or ambiguous prolonged occlusion may receive a **new ID**. Consequently different IDs across times/videos are not negative evidence. Positives use the same source-author fragment within one video; no independent physical reidentification is claimed. Duplicate frame/ID rows, source identity flags, duplicate/near-identical simultaneous boxes and class inconsistency quarantine the fragment; source per-frame/annotation quarantine is honored. This cohort had0source duplicate/class/reuse fragment quarantines. Native missing intermediate annotations are recorded, never interpolated; no missing-intermediate case occurred among accepted authentic pairs.

All160source hashes/full CPU decodes/pHash values were independently rechecked. Train/development source lineages are disjoint and had **0cross-split SHA/pHashHamming≤4 matches**. The frozen [15-lineage/2937-frame eval reservations](<F:/Code/HiFT/artifacts/wave3-training/FROZEN-001/reservations.json>) were reused hash-bound: no known eval exact/near content matches. Near pHash similarity **within one source video** is recorded as ordinary temporal similarity, not invented global identity. ETH05collection is held out from train; ADL09/10 are conservatively one TRAIN family, not independent collections. This is publisher/video/known-collection separation, **not proof of global physical-person or undiscovered re-edit disjointness**. No UAV/DTB eval boxes/flags are training labels.

## 2. Frozen indexed policy; conditional pilot deviation

[Policy](<F:/Code/HiFT/src/hift_research/temporal_contract/policy.py>) and eligible populations were frozen together with20input/transitive source-code hashes, source manifest, reservations, content receipt and seeds **before1024 attempts**.

- TRAIN indexed stream uses seed20261023+candidate_index. Its first draw is unchanged BernoulliPneg.05. Allowed stream lengths are1024preflight and1600futurepilot attempt slots; this is **not** authorization for SGD.
- PositiveP.95 candidates choose uniformly from the frozen ordered same-fragment endpoint population, both temporal directions, absolute native gap1..19. Endpoints are observed source frames, different native frame indices and different source image hashes, never interpolation or duplicated-frame positives.
- NegativeP.05 candidates choose uniformly from the **conditional simultaneous-known-distinct** population, not arbitrary cross-video objects. Both author tracks are present in the **same source frame** and source-qualified. No guessed temporal/cross-video ID relation is used.
- The negative eligible pool uses the **nominal actual287 search FOV**, with JPEG/interpolation support guards below. Actual TRAIN augmentation may invalidate nominal exclusion: rejection is retained, no alternative is drawn. Thus accepted prevalence may differ from5%.
- DEVELOPMENT seed20261024 has fixed128alternating classes, evenpositive/oddnegative,64each. Shift/scale/blur/flip/color are all0. It is a balanced loss-monitoring control, not the TRAIN negative distribution, trackingAUC or a benchmark-test tuning set.

These are deliberate **reduced temporal-pilot deviations**, not a reproduction of upstream four-dataset mixing, shuffled picks/repetition or arbitrary negative-video sampling. Missing VID/GOT/YTBB assets are not replaced by eval datasets. Source eligibility counts/exclusions are retained, including insufficient visibility(1285TRAIN/151DEV), unmarked/nonpresent(1572/73), and nominal-negative-footprint exclusions(4804/324). Sampling can naturally repeat source pairs across indices; zero replacement means no reject-driven refill, not sampling without replacement.

## 3. Negative absence proof and actual crop geometry

Distinct source boxes alone are insufficient: templateA can remain inside a search crop centered on objectB. The [pipeline](<F:/Code/HiFT/src/hift_research/temporal_contract/pipeline.py>) therefore records and composes the actual original-source→511preprocessing→127/287augmentation affine for both views. Preprocessing uses pinned SiamFC context/mean padding, JPEG95 encode/decode, original centered `_get_bbox` and unchanged pinned augmentation. Actual511 pixel scale is **510/source_crop_side**; source labels retain ideal127/contextscale, the disclosed511/510 mismatch. Images remain BGR, unnormalized CHW FP32. No source-resolution detail is created by resize.

For every negative:

1. Witness both source-author fragments simultaneously with raw/converted boxes, GT row IDs, flags and visibility.
2. Read the **actual** audited search `_crop_roi`, not an estimated object-box crop.
3. Expand that ROI by **18pixels in pre-JPEG511 coordinates** (16-pixel JPEG/chroma MCU plus2interpolation), clip to511pixel domain, invert the actual510-based preprocessing affine, then expand by **2original-source pixels**.
4. Require zero intersection between this guarded footprint and templateA's visible native source bbox. Overlap returns `template_target_visible_in_negative_search`, with source boxes, both augmented boxes, composed affines, footprint and overlap retained. No model-ready arrays are supplied for that rejection.
5. Accepted negatives have dense-zero cls1/cls2/offsets and zero regression weights. Positives require strictly positive geometric original-source support and nonempty target masks.

The guard is a declared conservative engineering support model under source annotations, not a semantic segmentation certificate or independent proof that the publisher never misboxed an object. Blur occurs after cropping; it does not create a larger pre-crop footprint. The one observed TRAIN rejection demonstrates this actual-footprint gate is live; its full geometry is retained in the supplement. No rejected attempt was replaced or relabeled.

Every accepted archive has all7array keys, checked shapes/dtypes/domains/finite values and dtype+shape+byte digests. Both views' original/centered/augmented boxes, JPEG hashes, affine inverses, source support and final array hashes are recorded. A fresh pipeline independently reproduced **every1024TRAIN and128DEV record and every1151accepted stored archive**, not just selected replay indices.

## 4. Measured distributions (973accepted TRAIN positives)

Marginal candidate-weighted distributions, not independent-object population estimates:

| Quantity | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Native temporal gap(frames) | 1 | 6 | 15 | 19 |
| Native gap(seconds) | .0333 | .2000 | .5000 | .6333 |
| Native center motion(px) | 0 | 10.77 | 70.33 | 125.00 |
| Motion /sqrt(templatearea) | 0 | .0933 | .8877 | 2.8035 |
| Search/template native area ratio | .5281 | 1 | 1.1554 | 1.7398 |
| Native aspect-ratio change | .5855 | 1 | 1.1272 | 1.8514 |
| Native bbox width(px) | 19 | 58 | 221.4 | 400 |
| Native bbox height(px) | 41 | 159 | 606.4 | 760 |
| Native bbox area(px²) | 798 | 9396 | 129948 | 285200 |
| Native bbox aspect(w/h) | .2110 | .3699 | .4799 | .5825 |
| Augmented label width(px) | 20.72 | 35.86 | 45.75 | 53.57 |
| Augmented label height(px) | 68.86 | 95.03 | 114.32 | 128.62 |
| Augmented label area(px²) | 2349.34 | 3364.30 | 4453.42 | 5303.00 |
| Regression active cells/pair | 12 | 22 | 34 | 52 |
| cls1positive cells/pair | 4 | 12 | 15 | 15 |
| cls1background cells/pair | 10 | 30 | 37 | 37 |
| cls1ignored cells/pair | 69 | 79 | 100 | 107 |
| cls2positive cells/pair | 16 | 34 | 56 | 80 |

Native visible fraction min0.52093, mean0.99592; actual original-image target support min798px². All accepted labeled search bboxes were contained in the287output (fraction1). Source motion/scale diagnostics are **before GT-centered training crops**: those crops recenter/normalize each frame just as pinned preprocessing does. These numbers are not model predictions or evidence that native displacement survives unchanged as crop-space translation.

**All active regression cells:**23363; minimum4.1667e-7, median0.50000039, P951.5, P99/max64. **298weight64cells** (1.2755% of active cells), across98pairs. This finite source heuristic is retained without retuning; optimization effects are unmeasured here. Across all1023accepted maps, masks total123783cells:11026cls1positive,33448background(includes6050negative-pair cells),79309ignored;36453cls2positive, dense cls2loss mask123783. Positive empty cls1/cls2/regression support counts are all**0**. Mask shape1×11×11; offsets4×11×11. Audit dtypes preserve source conventions; collator explicitly makes CPUFP32 tensors.

Actual TRAIN search crop511 ROI width ranged234.53..337.44px (median287.56,P95333.65); DEV width is exactly286px. Full realized shift/scale distributions and template/search containment are in the supplement. Pinned probabilities and seeds reproduce color/blur pixels; those realized color/blur draws are not separately named as fields in the canonical record.

**Authentic examples present:** stationary native center index19,MOT17-02 frames126→130; >100px native motion index205,MOT17-10 frames141→127; edge-clipped index63,MOT17-10 frames127→132. **Not observed in this cohort:** native area<64px², >2×/<0.5× area-scale change, missing intermediate annotation. Those cases are tested with explicitly **synthetic source-content/GT fixtures**, not falsely advertised as authentic examples.

## 5. Production Dataset, timing, bounded caches and input-fault STOP

```python
from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset, reject_aware_collate
from hift_research.temporal_contract.pipeline import TemporalPairPipeline

# Production-facing indexed CPU bridge; this does not authorize optimization.
dataset = TemporalCandidateDataset(manifest_path, train_policy_path,
                                   split="train", seed=20261023, length=1600)
record, arrays = dataset.candidate(0)  # same as dataset[0]; rejection returns arrays=None
# reject_aware_collate([(record, arrays)]) retains identities/rejections, never refills.
```

[Dataset](<F:/Code/HiFT/src/hift_research/temporal_contract/torch_adapter.py>) serializes paths/split/seed/length and hashes only, clearing dynamic source ops/caches via `__getstate__`; lazy per-PID construction is Windows-spawn-safe. The [unchanged collator](<F:/Code/HiFT/src/hift_research/data_contract/torch_adapter.py>) returns model_inputs/accepted_records/rejected_records/candidate_count/replacement_count. All-rejected batches have `model_inputs=None`. WorkerC owns independent actual two-worker/fault/recovery controller evidence. Scoped NumPy RNG is serial per process, **not thread-safe**; exact asynchronous six-worker resume is not claimed.

Decode and prepared-crop LRUs each bound pixel payload to256MiB. TRAIN final decode267494400bytes/43entries/840evictions; crop267910146bytes/342entries/1116evictions. Bounds refer to ndarray payload, **not process RSS/Python allocator overhead**. Read-only cached pixels, deterministic eviction and cold/warm/observer parity are tested.

`TemporalPairPipeline(...,stage_observer=observer)` calls external `observer.record(name,seconds,cache_hit=bool)` for decode, crop+JPEG, augmentation, targets and validation. Timings never enter canonical records or draw RNG. Cache/skipped-decode hits explicitly report0elapsed/cache_hitTrue. This hook supports lead-owned profiling with the actual bounded cache, not an imaginary all-resident rate; it is not itself a speed claim.

On decode-cache MISS, source bytes are read **once**, SHAchecked, then **that same payload** is decoded and dimensions checked. Changed/missing/corrupt input raises OSError/RuntimeError and does **not** become a geometry rejection or hidden skip. Dataset exceptions carry candidate_index for the lifecycle's STOP/recovery receipt. Source-hash checks on lazy initialization and frozen reservations/content receipts are fail-closed. Cached pixels remain previously authenticated immutable data; no claim of continuous on-disk polling is made.

## 6. Preserved failure001, exact recovery002, tests and remaining limits

FROZEN/PREFLIGHT001 failed on first negative,index3: NumPy bool from geometric comparison was not JSON serializable. Three positive records and the orphan index3array were preserved, never counted as a completed cohort. The [failure receipt](<F:/Code/HiFT/artifacts/wave4-preflight/PREFLIGHT-001/failure_receipt.json>) is explicit. No `default=str`, overwrite, data/seed/pool tuning, or favorable resampling was used.

Only three native-Python scalar conversions were added. FROZEN/PREFLIGHT002 is a **new ID**. [All20failed001source files](<F:/Code/HiFT/artifacts/wave4-preflight/FAILED-001-SOURCE/snapshot_receipt.json>) were reconstructed from exact inverse edits and verified against001's precommitted SHA values. A separately labeled forensic manifest references those immutable snapshots, not a production hash bypass. [Forensic comparison](<F:/Code/HiFT/artifacts/wave4-preflight/FORENSIC-001/comparison.json>) re-executes exact old source: original first-negative JSON failure is reproduced, while indices0..2positive **and index3negative** numeric geometry, canonical field values and every archived array equal002 exactly. Semantic policy, source, population and seeds are unchanged.

[22CPUtests](<F:/Code/HiFT/tests/test_temporal_contract.py>) passed in13.11s after recovery, covering every128-record JSON roundtrip, stationary/motion/rapid-scale/tiny/clipped/fully-offimage synthetic content, missing-mid no interpolation, negative crop contradiction/rejection, native format/source GT/presence contradictions, IDreuse/duplicate ambiguity, source/eval collision, seed/split guards, pickle/collation, LRU/observer parity, same-sized media hash-change STOP, and reservation/content-receipt tampering. All existing Wave3frozen inputs/upstream remain unedited.

Immutable summaries record gitrevision `28e9e197c62a21450587465e4c32abdf97fb88d3`, tracked dirty-diff SHA, clean pinned upstream `992aa7c1605360d607af84cb804b93cd8ee81caf`, individually hashed input code, Python3.13.0/NumPy2.4.4/OpenCV4.13.0/torch2.10.0+cu128 distribution, Windows10,20logical CPU, and null model/checkpoint/optimizer. The20file input hashes—not merely tracked diff—bind new adapter sources.

Remaining limits: only3TRAIN clips/oneDEV collection, people/class1 only, visibility filtering and conditional negative-pool bias, source-author fragment/box quality rather than independently solved reID, unknown global persons/re-edits, repeated DEVnegative controls, no authentic extreme-scale/tiny/missing-mid example, no original four-dataset recipe, no convergence/tracking gain/reproduction evidence. Real-data SGD remains **unauthorized**. No R1/adaptive crop, benchmark retuning, official-checkpoint optimization, eval-asset training or real learned checkpoint was produced by this worker.
