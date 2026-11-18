# Per-layer int8 sensitivity of HiFT's backbone (DEV, 20 videos)

Identity with round 4: {'draw1_calibration_matches_round4': True, 'none_draw1_wakeboard7_boxes_iou>0.99': 1.0, 'keep1_draw1_wakeboard7_boxes_iou>0.99': 1.0, 'draw2_calibration_matches_round4': True, 'draw3_calibration_matches_round4': True}

## A. Score-map fidelity vs fp32 (mean over draws)

192 DEV (template, search) pairs; same-peak = share of crops where the int8 model's score-map peak is at the same cell as fp32; rel-err = relative L2 error of the fused score map. fp32 network latency [88.98, 86.73] ms (start, end).

| Config | Layers kept fp32 | Same-peak | Rel. error | Network ms |
|---|---|---|---|---|
| none | - | 0.955 | 0.0148 | 40.6 |
| keep1 | layer1 | 0.976 | 0.0091 | 49.6 |
| keep2 | layer2 | 0.948 | 0.0139 | 54.7 |
| keep3 | layer3 | 0.948 | 0.0144 | 46.1 |
| keep4 | layer4 | 0.957 | 0.0146 | 47.9 |
| keep5 | layer5 | 0.953 | 0.0148 | 44.9 |
| only1 | layer2+layer3+layer4+layer5 | 0.962 | 0.0130 | 69.6 |
| only2 | layer1+layer3+layer4+layer5 | 0.983 | 0.0082 | 64.7 |
| only3 | layer1+layer2+layer4+layer5 | 0.988 | 0.0037 | 79.9 |
| only4 | layer1+layer2+layer3+layer5 | 0.995 | 0.0022 | 78.6 |
| only5 | layer1+layer2+layer3+layer4 | 0.983 | 0.0036 | 81.3 |

**Q1. Most sensitive single layer (largest error when quantised alone):** layer1 (ranking: layer1, layer2, layer3, layer5, layer4). Layer1 is the most sensitive.

## B. DEV tracking: change in success AUC vs fp32 CPU (per video: mean over the available calibration draws)

| Config | Layers kept fp32 | Draws | Change [95% CI] | Excluding person19_2 | ms/frame |
|---|---|---|---|---|---|
| none | - | 3 | +0.0197 [-0.0102, +0.0692] | -0.0031 [-0.0129, +0.0030] | 42.3 |
| keep1 | layer1 | 3 | +0.0251 [-0.0035, +0.0738] | +0.0024 [-0.0057, +0.0110] | 51.1 |
| keep2 | layer2 | 3 | +0.0464 [-0.0057, +0.1194] | +0.0250 [-0.0099, +0.0833] | 56.1 |
| keep3 | layer3 | 3 | +0.0213 [-0.0051, +0.0585] | +0.0064 [-0.0081, +0.0276] | 48.5 |
| keep4 | layer4 | 3 | +0.0307 [-0.0032, +0.0793] | +0.0164 [-0.0048, +0.0543] | 50.3 |
| keep5 | layer5 | 3 | +0.0206 [-0.0069, +0.0578] | +0.0058 [-0.0111, +0.0279] | 47.3 |
