## Tracking

| run | variant | attack | eps | defense | adaptive | success_auc | precision_20 | fps_end_to_end |
|---|---|---|---|---|---|---|---|---|
| uav123-sub40_fp32_fgsm1_nodef | fp32 | fgsm | 1.000 | none | False | 0.552 | 0.772 | 33.063 |
| uav123-sub40_fp32_fgsm2_nodef | fp32 | fgsm | 2.000 | none | False | 0.473 | 0.709 | 32.883 |
| uav123-sub40_fp32_fgsm4_nodef | fp32 | fgsm | 4.000 | none | False | 0.418 | 0.672 | 32.882 |
| uav123-sub40_fp32_fgsm8_nodef | fp32 | fgsm | 8.000 | none | False | 0.319 | 0.578 | 32.536 |
| uav123-sub40_fp32_noise2_nodef | fp32 | noise | 2.000 | none | False | 0.582 | 0.756 | 70.201 |
| uav123-sub40_fp32_noise4_nodef | fp32 | noise | 4.000 | none | False | 0.563 | 0.729 | 70.757 |
| uav123-sub40_fp32_noise8_nodef | fp32 | noise | 8.000 | none | False | 0.556 | 0.738 | 70.031 |
| uav123-sub40_fp32_clean_nodef | fp32 | none | 0 | none | False | 0.582 | 0.754 | 93.942 |
| uav123_fp32_clean_nodef | fp32 | none | 0 | none | False | 0.584 | 0.754 | 95.359 |
| uav123_fp32_clean_nodef_t4 | fp32 | none | 0 | none | False | 0.586 | 0.758 | 10.849 |
| uav123-sub40_fp32_pgd4s10_nodef | fp32 | pgd | 4.000 | none | False | 0.095 | 0.323 | 6.291 |

## Edge cost

| variant | device | threads | size_mb | gflops_total | latency_ms_mean | fps_network | same_peak_fraction |
|---|---|---|---|---|---|---|---|
| fp32 | cpu | 1 | 44.333 | 13.208 | 177.515 | 5.633 | 1.000 |
| int8bb | cpu | 1 | 33.095 | 13.208 | 75.170 | 13.303 | 0.933 |
| int8dyn | cpu | 1 | 37.260 | 13.208 | 167.025 | 5.987 | 1.000 |
| prune50 | cpu | 1 | 44.333 | 13.208 | 172.140 | 5.809 | 0.550 |
| fp32 | cpu | 2 | 44.333 | 13.208 | 93.575 | 10.687 | 1.000 |
| int8bb | cpu | 2 | 33.095 | 13.208 | 43.939 | 22.759 | 0.933 |
| int8dyn | cpu | 2 | 37.260 | 13.208 | 91.880 | 10.884 | 1.000 |
| prune50 | cpu | 2 | 44.333 | 13.208 | 93.076 | 10.744 | 0.550 |
| fp32 | cpu | 4 | 44.333 | 13.208 | 89.434 | 11.181 | 1.000 |
| int8bb | cpu | 4 | 33.095 | 13.208 | 42.715 | 23.411 | 0.933 |
| int8dyn | cpu | 4 | 37.260 | 13.208 | 87.985 | 11.366 | 1.000 |
| prune50 | cpu | 4 | 44.333 | 13.208 | 89.194 | 11.211 | 0.550 |
