# Matched re-timing: fp32 vs the accuracy-tested int8 model

- CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz (host TC2N01), torch 2.8.0+cu128
- int8 model: int8bb-s1, calibration 40 videos x 4 (spread), set 1
- Same model as the accuracy run? calibration videos identical: **True**
- 6 sessions per thread count, order alternating; tracker step on bike1, car1_2, person11, person7_2 (5496 frames per model per session; image loading excluded)

| Threads | fp32 network ms | int8 network ms | Network speed-up | fp32 step ms | int8 step ms | Tracker-step speed-up |
|---|---|---|---|---|---|---|
| 4 | 89.1 | 50.9 | **1.75×** (1.74–1.76) | 90.5 | 52.5 | **1.72×** (1.71–1.73) |

Values are medians over sessions; speed-ups are within-session ratios (median, range).
Model size: fp32 44.3 MB, int8 33.2 MB.

Identity/consistency check (share of frames whose box matches the saved accuracy run at IoU > 0.99):
- fp32_session1: bike1 1.000, car1_2 1.000, person11 1.000, person7_2 1.000
- int8bb-s1_session1: bike1 1.000, car1_2 1.000, person11 1.000, person7_2 1.000
- int8bb-s1_session2: bike1 1.000, car1_2 1.000, person11 1.000, person7_2 1.000
- fp32_session2: bike1 1.000, car1_2 1.000, person11 1.000, person7_2 1.000
