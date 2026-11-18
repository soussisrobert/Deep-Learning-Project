# selection_log.md: settings tried before gamma 0.6 and the anchor

Source: `experiments_log.md` (entry names E0-E10, logbook numbers in the last column of the logbook where the logbook entry exists; numbers are mean success AUC on the stated sample). All selection happened on NAT2021 videos (or on subsets drawn from NAT2021). NAT2021L was used once, to confirm gamma 0.6 (E9). UAVDark135 and UAV123 were not used to choose gamma or the anchor; first.pth was evaluated on them only after both were fixed (fine-tuning phase, entries 68-71).

"Found" = the result that was recorded. `GAP:` = not recorded or not run.

| # | Setting tried | Benchmark / sample | Found (success AUC) | Entry |
|---|---|---|---|---|
| 1 | baseline, no enhancement | NAT2021 180 | 0.372 (precision 0.553) | E0 |
| 1b | baseline | 30-video subset (every 6th, `--every 6`) | 0.286 | E1 |
| 1c | baseline | balanced 40-video sample (20 improved-half + 20 worse-half videos of the gamma 0.4 run, fixed seed) | 0.380 | E3c |
| 2 | gamma 0.4 both crops | 30 subset | 0.350 | E1 |
| 2b | gamma 0.4 both crops | 180 | 0.398 | E2 |
| 3 | CLAHE (clip 3.0, 8x8) both crops | 30 / 180 | 0.350 / 0.384 | E1 / E2 |
| 4 | gamma 0.4 + CLAHE clip 3.0, both crops | 30 / 180 | 0.344 / 0.388 | E1 / E2 |
| 5 | gamma + CLAHE on the search crop only | 30 | 0.301 | E1 |
| 6 | gamma + CLAHE on the template crop only | 30 | 0.246 (below baseline 0.286) | E1 |
| 7 | gamma / CLAHE-clip pairs (0.6/2.0, 0.5/2.0, 0.45/3.0, 0.5/4.0, control 1.0/1.0) on the 60 videos where gamma 0.4 was worse | 60 videos | GAP: cancelled by the user before completing, no result recorded | E3a (cancelled) |
| 8 | gamma 0.95 / 0.9 / 0.8 / 0.6 / 0.5 / 0.4 both crops | balanced 40-video sample | 0.397 / 0.404 / 0.452 / 0.452 / 0.429 / 0.438 | E3c |
| 9 | CLAHE clip 1 / 2 / 3 / 4 both crops | balanced 40-video sample | 0.418 / 0.432 / 0.415 / 0.387 | E3c |
| 10 | gamma 0.8 / 0.6 both crops | 180 | 0.402 / **0.417** (gamma 0.4: 0.398) | E4 |
| 11 | denoise (Gaussian 3x3 / Gaussian 5x5 / bilateral) then gamma 0.6 | balanced 40-video sample | 0.439 / 0.438 / 0.452 (gamma 0.6 alone 0.452) | E5 |
| 12 | denoise with non-local means then gamma 0.6 | balanced 40 | GAP: dropped, 4.1 fps (about 25x slower), not evaluated | E5 |
| 13 | gamma 0.6 on NAT2021L (confirmation) | NAT2021L 23 | 0.316 (baseline 0.299; within noise) | E9 |
| 14 | template rebuilt from the last 3 frames, pixel average / feature average, no anchor | balanced 40 (gamma 0.6) | 0.331 / 0.334 (frame-1 template 0.452) | E7 |
| 15 | anchor weight 0.5 / 0.2 (feature average of last 3), no gate | balanced 40 (gamma 0.6) | 0.434 / **0.484** | E8 |
| 16 | anchor 0.5 + gate / anchor 0.2 + gate / gate only (median-score gate) | balanced 40 | 0.475 / 0.463 / 0.402 | E8 |
| 17 | anchor 0.2 + gamma 0.6 | NAT2021 180 (confirmation) | **0.453** (gamma 0.6 alone 0.417; +0.036 [+0.018, +0.055]); on the 140 videos not in the 40-sample +0.038 | E10 |
| 18 | gamma values other than 0.4 / 0.5 / 0.6 / 0.8 / 0.9 / 0.95 (for example 0.7) | -- | GAP: not tried; on the 180 set only 0.4, 0.6, 0.8 were run  | -- |
| 19 | anchor weights other than 0.5 / 0.2 (for example 0.1, 0.3) and averages over a number of frames other than 3 | -- | GAP: not tried (the last-3-frames choice came from the user's idea P15 and was not varied)  | -- |
| 20 | gamma on the template only / search only | 30 (gamma + CLAHE only) | GAP: only the combined gamma+CLAHE variants were run one-sided (rows 5-6); gamma alone one-sided was not run  | -- |

Which benchmark chose what
- **Gamma 0.6:** chosen on NAT2021 videos: first screened on the 30-video subset (E1, gamma 0.4 only), then on the balanced 40-video sample (E3c, six gamma values) and all 180 videos (E4, three values). The final value 0.6 is the best of the three tested on all 180 NAT2021 videos, which are also the headline benchmark, so the NAT2021 number for gamma is selection-biased (E4 says so). NAT2021L (E9) gave the same direction but within noise.
- **Anchor 0.2, last 3 frames:** chosen on the 40-video sample drawn from NAT2021 (E8, five variants, best 0.484, interval against gamma alone [-0.020, +0.084], not distinguishable), then confirmed on all 180 (E10). The 140 videos outside the selection sample gave +0.038, so the gain is not only selection.
- **Anchor on other sets:** anchor + gamma was first measured on NAT2021L, UAVDark135 and UAV123 in the fine-tuning phase (E16 / E19), after it was fixed.
- **Not used for selection:** UAVDark135 test, UAV123 held-out day. UAVDark135 val was used for fine-tuning epoch selection and E21.
