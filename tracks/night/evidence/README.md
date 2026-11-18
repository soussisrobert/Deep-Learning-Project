# night_evidence/ : evidence package for the night track (built 2026-11-17, logbook entry 90, job 55475)

Everything here comes from existing result folders and the logbook (`experiments_log.md`); no tracking was run for this package. Where something could not be provided the text says `GAP: ...`.

## Files
| File | Content |
|---|---|
| `protocol.md` | the protocol template, filled; official-setup check; leakage paragraph |
| `claims.md` | one line per claim: Claim / Source file / Selection caveat |
| `code_ref.md` | repo state and the exact files / lines for gamma and the anchored update; scripts that produced the numbers |
| `selection_log.md` | every setting tried before gamma 0.6 and the anchor, with benchmark, sample and logbook entry |
| `per_video_NAT2021.csv`, `per_video_NAT2021L.csv`, `per_video_UAVDark135_test.csv`, `per_video_UAVDark135_val.csv` | one row per video per model (original, gamma, gamma_anchor), columns `benchmark,video,frames,model,seed,success_auc,precision20,toolkit` |
| `official_vs_own_check.csv`, `absent_gt_handling.csv` | cross-checks of the per-video numbers and the effect of absent-GT frames on precision |
| `E26_speed_summary.txt`, `E26_defaults_cfg_per_video.csv` | speed of the three settings on one A40 (job 55514, two passes) and the NAT2021L per-video AUCs with the `config.py` default tracking values (claims 16 and 17) |
| `figures/` | `fig_headline_bars.png`, five before/after strips (`strip_*.png`), `strips_index.csv`, `E23_package.py` (the script that made the CSVs and figures; the original is `tools/illumination/E23_package.py`) |

## Headline (mean success AUC, first.pth, test-time settings only)
| Benchmark (videos) | original | gamma 0.6 | gamma 0.6 + anchor 0.2 |
|---|---|---|---|
| NAT2021 (180) | 0.372 | 0.417 | 0.453 |
| NAT2021L (23) | 0.299 | 0.316 | 0.377 |
| UAVDark135 test (23) | 0.352 | 0.434 | 0.443 |
| UAVDark135 val (16) | 0.422 | 0.475 | 0.493 |

Computed from the per-video CSVs (official pysot OPE code, see `protocol.md`). Precision@20 px (official code) for the same rows: NAT2021 0.553 / 0.606 / 0.675; NAT2021L 0.446 / 0.445 / 0.536; UD135 test 0.452 / 0.566 / 0.585; UD135 val 0.518 / 0.599 / 0.624. Official precision counts frames with an absent GT as hits (see `protocol.md`, "absent GT").

## Fine-tuning runs (all start from first.pth; 10 epochs of 330,000 pairs, heads LR 2e-4 -> 2e-5 and backbone layers 3-5 at 0.1x that; checkpoint per run selected by tracking success AUC on the UAVDark135 val split, anchor + gamma curve, logbook entries 68-78)
| Run | Training data / augmentation | Selected epoch |
|---|---|---|
| A | mix of sampled pairs 0.4 UAVDark135 train split / 0.3 UAVDT / 0.3 UAV123 (UAVDT and UAV123 darkened by the night augmentation), no random enhancement (ENH_P 0), no negatives | 8 |
| B | A + gamma augmentation (probability 0.5, one gamma shared by template and search, U[0.4, 1.0]) | 8 |
| C | B + real negatives (NEG 0.2, HARDNEG 0.1) | 10 |
| D | control: daytime data only (UAV123 undarkened, UAVDT daylight tracks), no darkening, no enhancement, no negatives | 1 |

Splits (entry 64): UAVDark135 scene-aware split by whole linked group, 75 / 10 / 15 % of frames (train / val / test); UAV123 held-out daytime test = sequences with prefix `group`, `wakeboard`, `boat`; DarkTrack2021 dropped (100 of 110 sequences are NAT2021 footage, entry 62). Exact data lists per run: `experiments/night_ft/run_{A,B,C,D}.yaml`.

Numbers behind "ties within 0.013" (E16, mean success AUC difference run minus first.pth, setting anchor 0.2 + gamma 0.6; source `tools/illumination/E16_ablation_bootstrap.csv`):
| Set | A | B | C | D |
|---|---|---|---|---|
| NAT2021 (180) | -0.012 | -0.004 | -0.009 | -0.013 |
| NAT2021L (23) | -0.013 | +0.005 | +0.010 | -0.026 |
| UAVDark135 test (23) | -0.004 | +0.018 | +0.045 | -0.041 |
| UAV123 held-out day (30) | -0.002 | -0.018 | -0.003 | +0.005 |

"All runs within 0.013 of first.pth" holds on NAT2021 only (largest |difference| 0.0129, D); on the other three sets at least one run is further than 0.013 away (NAT2021L D -0.026, UD135 test C +0.045 and D -0.041, UAV123 B -0.018). On NAT2021 all four paired bootstrap intervals include 0 (E16). The epoch of each run was chosen on UAVDark135 val, not on NAT2021.

## sources/ (E29): copies of the files cited in claims.md
`sources/` holds 27 files (0.78 MB), copies of the originals with user path, username and cluster / partition names removed (only `output_E22_A_to_C.out` needed it: a `du` line). Claims 1-17: `E15_stretches.csv`, `E15_stretch_length_hist.csv`, `output_E15_lost_diag.out` (claims 9-10); `E17_roc_auc.csv`, `E17_template_sim.csv`, `output_E17_template_sim.out` (claim 11); `E22_A_start_categories.csv`, `E22_A_lost_frame_peaks.csv`, `output_E22_A_to_C.out` (claim 12); `E16_ablation_bootstrap.csv` (claims 13-14b); `E18_compare.csv`, `E18_summary.csv`, `output_E18_bank.out` (claim 15); `E19_settings_bootstrap.csv` (claims 5-8); `speed_{original,gamma,gamma_anchor}_p{1,2}.log` (claim 16, per-video Time lines kept). Split and leakage evidence for `protocol.md`: `output_HiFT_uavdark_leak_49103.out` (overlap check, job 49103), `leak_UAVDark135.csv`, `leak_UAVDark135_within.csv` (frame-pair correlations, against NAT2021 / NAT2021L and inside UAVDark135), and the split manifest `UAVDark135_train.txt`, `UAVDark135_val.txt`, `UAVDark135_test.txt` plus `UAVDark135_groups.csv` (linked groups, frames, mean gray and the split each group went to; made by `tools/p1_plan.py` from the same-footage pairs of `tools/leak_one.py`). Claims 1-4 and 15-17 also cite files that are already in this folder (`per_video_*.csv`, `official_vs_own_check.csv`, `absent_gt_handling.csv`, `E26_*`). The code behind the numbers (`E22_loss_taxonomy.py`, `lost_diag.py`, `template_sim.py`, `test_bank.py`, `hift_tracker_bank.py`, `E19_settings_bootstrap.py`, `eval_nat.py`, `enhance.py`, `hift_tracker_tmpl.py`) is on branch `night-full`; `f28900cf`, `b41f9359` and `255a9fbb` are ancestors of it.

### E26 timer (read-only facts)
- `tools/test.py`: `for idx, (img, gt_bbox) in enumerate(video):` line 96, `tic = cv2.getTickCount()` line 97, `toc += cv2.getTickCount() - tic` line 125, `toc /= cv2.getTickFrequency()` line 142, printed as `Time: {}s Speed: {}fps` with `idx / toc` (lines 152-153). `tools/test_tmpl.py`: the same, lines 106, 107, 135, 152, 162-163.
- The image is read by the iterator (`toolkit/datasets/video.py` `__getitem__`: `cv2.imread(self.img_names[idx])`) when the `for` statement fetches the next item, before line 97, so **image reading is outside the timer**. The timer covers `tracker.init` / `tracker.track` and the loop body up to line 125 (also the `--reinit_every` block, off by default); it does not cover writing the result file. The printed speed is `idx / toc` with `idx` the last frame index, i.e. (frames - 1) / seconds.

### Frames evaluated, stride and frame rate (read-only facts)
- **Every frame is used.** `--every` (`test.py` line 49, `test_tmpl.py` line 57, default 1; `eval_nat.py` the same) selects VIDEOS: `if v_idx % args.every != 0: continue` (`test.py` line 83, `test_tmpl.py` line 93). Inside a video every frame is tracked (`enumerate(video)`); there is no frame stride. All reported numbers use the default (all videos); the speed job (55514) used `--every 9`, i.e. every 9th video (20 of 180), all frames of those videos.
- Frames evaluated (all frames, absent-GT frames included as failures; from the E20 run): NAT2021 140,815 frames (180 videos; 3,036 without a usable GT), NAT2021L 53,564 (23; 948), UAVDark135 test 18,865 (23; 255), UAVDark135 val 12,638 (16; 234). The first frame of each video is the initialisation, so tracked frames are those numbers minus the number of videos.
- **Frame rate: not recorded.** The dataset files (`NAT2021.json`, `NAT2021L.json`, `NATUD135TEST.json`, `NATUD135VAL.json`) hold only `video_dir`, `init_rect`, `img_names` (and the annotations); no field named fps / frame rate exists, and the dataset folders contain no file for it. The raw UAVDark135 folder is not on disk, so its frame rate could not be checked.
