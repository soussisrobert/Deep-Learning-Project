# Round 2: results against the pre-registered criteria

Baseline on the 40 sequences: clean **0.582**, PGD-10 ε=4 **0.095** (lost 0.487). Clean tracking time 10.64 ms/frame (no image loading).

## 1. Defences (proposed improvement for robustness)

Pass = clean cost ≥ −0.010 AND adaptive gain ≥ +0.10 with interval above 0 AND time overhead ≤ 3.9%.

| Defence | Clean AUC (Δ vs clean) | PGD, attacker unaware (Δ vs PGD) | PGD, adaptive (Δ vs PGD) | Share of loss recovered (adaptive) | Defence time per frame (share of clean) | Verdict |
|---|---|---|---|---|---|---|
| blur1 | 0.543 (-0.040 [-0.078, -0.011] (n=40, +0/-6)) | 0.394 (+0.299 [+0.224, +0.373] (n=40, +34/-1)) | 0.210 (+0.114 [+0.072, +0.159] (n=40, +17/-0)) | 23% | +0.24 ms (+2.3%) | robust but costly |
| median3 | 0.555 (-0.027 [-0.064, -0.002] (n=40, +0/-3)) | 0.258 (+0.163 [+0.112, +0.219] (n=40, +25/-0)) | 0.179 (+0.083 [+0.050, +0.122] (n=40, +17/-0)) | 17% | +1.23 ms (+11.6%) | partial adaptive gain (below +0.10) |
| bits5 | 0.556 (-0.026 [-0.062, -0.001] (n=40, +0/-3)) | 0.114 (+0.018 [+0.003, +0.039] (n=40, +2/-1)) | 0.122 (+0.027 [+0.010, +0.049] (n=40, +5/-0)) | 5% | +0.17 ms (+1.6%) | partial adaptive gain (below +0.10) |
| jpeg75 | 0.571 (-0.012 [-0.029, +0.000] (n=40, +0/-2)) | 0.413 (+0.317 [+0.255, +0.382] (n=40, +34/-0)) | 0.355 (+0.259 [+0.200, +0.320] (n=40, +34/-0)) | 53% | +1.46 ms (+13.7%) | robust but costly |

## 2. Compression (proposed improvement for efficiency)

Pass = network ≥ 1.5× faster than fp32 at 4 CPU threads AND accuracy drop ≤ 0.010 (interval reported).

| Variant | Network ms (4 thr) | Speed-up | Size MB | AUC on 40 (Δ vs fp32 CPU) | AUC on 123 (Δ vs fp32 CPU) | PGD ε=4, fp32 attacker (Δ vs fp32 PGD) | Verdict |
|---|---|---|---|---|---|---|---|
| int8bb | 41.2 | 2.12× | 33.1 | 0.576 (-0.012 [-0.027, -0.001] (n=40, +0/-3)) | 0.587 (+0.000 [-0.009, +0.011] (n=123, +2/-4)) | 0.101 (+0.006 [-0.005, +0.019] (n=40, +3/-2)) | faster but less accurate |
| int8full | 37.2 | 2.34× | 21.8 | 0.519 (-0.069 [-0.109, -0.035] (n=40, +0/-12)) | 0.546 (-0.041 [-0.064, -0.018] (n=123, +3/-22)) | 0.097 (+0.002 [-0.008, +0.014] (n=40, +2/-3)) | faster but less accurate |
| prune50 | 89.2 | 0.98× | 44.3 | 0.459 (-0.130 [-0.177, -0.085] (n=40, +3/-27)) | not run yet | not run yet | not faster |
| int8dyn | 88.0 | 0.99× | 37.3 | not run yet | not run yet | not run yet | not faster |

fp32 CPU reference on the same 40 sequences: 0.588 (all 123: 0.586).

## 3. Attack strength and stability

| Run | Success AUC | Δ vs clean |
|---|---|---|
| fp32_noise1_nodef | 0.582 | -0.001 [-0.015, +0.013] (n=40, +1/-1) |
| fp32_fgsm1_nodef | 0.552 | -0.030 [-0.084, +0.028] (n=40, +9/-20) |
| fp32_pgd1s10_nodef | 0.504 | -0.078 [-0.146, -0.008] (n=40, +9/-24) |
| fp32_pgd2s10_nodef | 0.279 | -0.303 [-0.387, -0.214] (n=40, +4/-31) |
| fp32_pgd4s10_nodef | 0.095 | -0.487 [-0.574, -0.392] (n=40, +1/-35) |
| fp32_pgd8s10_nodef | 0.080 | -0.502 [-0.588, -0.410] (n=40, +1/-35) |
| fp32_pgd4s10_nodef_seed1 | 0.102 | -0.481 [-0.567, -0.387] (n=40, +1/-35) |
| fp32_pgd4s10_nodef_seed2 | 0.110 | -0.473 [-0.559, -0.380] (n=40, +1/-35) |
| fp32_noise4_nodef | 0.563 | -0.020 [-0.057, +0.005] (n=40, +1/-2) |
| fp32_noise4_nodef_seed1 | 0.563 | -0.019 [-0.057, +0.006] (n=40, +1/-2) |
| fp32_noise4_nodef_seed2 | 0.570 | -0.013 [-0.047, +0.010] (n=40, +1/-1) |

pgd4s10 over 3 seeds: mean 0.102, SD 0.007 → stable

noise4 over 3 seeds: mean 0.565, SD 0.004 → stable
