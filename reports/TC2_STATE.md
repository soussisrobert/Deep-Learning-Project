# HiFT on TC2 — state log

Account ning011 (shared with a PrismKey session). All HiFT files under `/home/msai/ning011/hift`.
Rules: F:\Code\prismkey-omr-research-data\tc2_stage\TC2_GUIDE.md + official CCDS TC2 user guide.

| Time (UTC+8) | Action | Detail | Result |
|---|---|---|---|
| 2026-10-28 | home usage (read-only, home only) | `du` in /home/msai/ning011 after owner cleanup | 51 GB used of 100 GB quota |
| 2026-10-28 | bundle built | commit ee80b6781f96, hift_code.tar 119 MB sha256 ee2e5ee3…e8c1 | LF-verified |
| 2026-10-28 01:50 | upload | ~/hift/upload (tar, sha256 files, GIT_COMMIT, installer) | sha256 OK on TC2 |
| 2026-10-28 | install env-data | unpacked ~/hift/repo_ee80b6781f96 (payload sha256 OK) | submitted hift_env=48129, hift_data=48130 (afterok:48129), TC2N03 |

## Planned run queue (rolling: next submitted only when the previous job is done and the running one is half way)
hift_b0ft_orig (~1.5 h) → hift_b0ft_corr (~1.5 h) → hift_s3_alex18 (~2 h) → hift_s3_alex40 (~2 h) → hift_s3_res18 (~3.5 h) → hift_s3_res40 (~3.5 h)

## Earlier rule breaches (2026-10-27, before the guide was provided)
SSH probes without the -O check guard; `df` on /home/msai, /, /scratch, /tmp; on a compute node `df /tmp /dev/shm`
and `ls -d /scratch* /local*`; python/curl on the head node; one 3-min `srun` probe job on TC2N03.

## Orchestrator log (appended automatically)
| Time (UTC+8) | Action | Detail | Result |
|---|---|---|---|
| 2026-10-28 02:06 | upload | bundle d798648 (sha256 1bce281b…) + resnet18 weights (f37072fd…) | sha256 OK, unpacked |
| 2026-10-28 02:08 | upload | bundle 46a99aa (sha256 f9c2345f…), set -u fix | sha256 OK, unpacked |
| 2026-10-28 02:08 | submit env | hift_env=48138; data refused | will submit data after env completes |
| 2026-10-28 06:42 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 07:06 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 07:30 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 07:53 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 08:17 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 08:41 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 09:04 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 09:28 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 09:52 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 10:15 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 10:39 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 11:02 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 11:26 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 11:49 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 12:13 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-28 12:20 | note | hift_env 48138 COMPLETED in 294 s (~/hift/env ready). Rows "socket check failed" 06:42-12:13 come from an orchestrator bug (guard exit inside a subshell did not stop the main loop; only local `ssh -O check` calls ran, nothing reached TC2). Fixed (USR1 to main shell). Data job never submitted: stuck hift_data 48130 (DependencyNeverSatisfied) still holds a slot. | orchestrator stopped; waiting for owner: reconnect socket + scancel 48130 |
| 2026-10-28 12:21 | submit data | hift_data2=48265 (repo hift/repo_46a99aa5273a) | submitted |
| 2026-10-28 13:03 | data | 48265|hift_data2|FAILED|328 | FAILED - stopping |
| 2026-10-28 13:06 | upload | bundle 953d6cd (sha256 73ddd032…), range-read retry + CRC resume | sha256 OK, unpacked |
| 2026-10-28 13:06 | submit data | hift_data2=48278 (repo hift/repo_953d6cd58976) | submitted |
| 2026-10-28 13:48 | data | hift_data2=48278 | COMPLETED |
| 2026-10-28 13:48 | submit hift_b0ft_orig | run TC2-B0FT-ORIG-001 | job 48295 |
| 2026-10-28 14:57 | scancel | 48130 hift_data (own stuck job, DependencyNeverSatisfied) | cancelled on explicit owner instruction (chat: "just scncel it") |
| 2026-10-28 15:25 | retrieve TC2-B0FT-ORIG-001 | final-model.pt sha256 bb88f1f841602a27… + summary/manifest/journal/slurm log | OK |
| 2026-10-28 15:32 | eval TC2-B0FT-ORIG-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.585198346 precision@20px=0.771652843; partial |
| 2026-10-28 15:47 | eval TC2-B0FT-ORIG-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.585070056 precision@20px=0.782989031; expected_count_only |
| 2026-10-28 15:52 | eval TC2-B0FT-ORIG-001 | UAV123, raw, nan-safe | sequence_macro success_auc=0.573932023 precision@20px=0.756024176; partial |
| 2026-10-28 16:04 | eval TC2-B0FT-ORIG-001 | VISDRONE, raw, nan-safe | sequence_macro success_auc=0.584420083 precision@20px=0.779195271; expected_count_only |
| 2026-10-28 16:04 | submit hift_b0ft_corr | run TC2-B0FT-CORR-001 | job 48352 |
| 2026-10-28 16:54 | submit hift_s3_alex18 | run TC2-S3-ALEX-SC18-001 | job 48382 |
| 2026-10-28 18:15 | orchestrator v5 | 15-min checks, background eval, 1 retry per job (new run id) | started; alex18 48382 FAILED (no Python exception in journal; batch killed externally) -> retry as TC2-S3-ALEX-SC18-002 |
| 2026-10-28 18:15 | hift_b0ft_corr | 48352 TC2-B0FT-CORR-001 | COMPLETED 4344s |
| 2026-10-28 18:15 | retrieve TC2-B0FT-CORR-001 | final-model.pt sha256 2c63c13cd3da5eec... + summary/manifest/journal/slurm log | OK |
| 2026-10-28 18:15 | submit hift_s3_alex18 | run TC2-S3-ALEX-SC18-002 | job 48432 |
| 2026-10-28 18:20 | eval TC2-B0FT-CORR-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.588070494 precision@20px=0.763281837; partial |
| 2026-10-28 18:39 | eval TC2-B0FT-CORR-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.613094890 precision@20px=0.829939766; expected_count_only |
| 2026-10-28 18:50 | eval TC2-B0FT-CORR-001 | UAV123, raw, nan-safe | sequence_macro success_auc=0.572068782 precision@20px=0.740625512; partial |
| 2026-10-28 19:14 | eval TC2-B0FT-CORR-001 | VISDRONE, raw, nan-safe | sequence_macro success_auc=0.567051419 precision@20px=0.763358433; expected_count_only |
| 2026-10-28 19:30 | submit hift_s3_alex40 | run TC2-S3-ALEX-SC40-001 | job 48453 |
| 2026-10-28 20:45 | hift_s3_alex18 | 48432 TC2-S3-ALEX-SC18-002 | COMPLETED 8734s |
| 2026-10-28 20:45 | retrieve TC2-S3-ALEX-SC18-002 | final-model.pt sha256 1ca14543834c6207... + summary/manifest/journal/slurm log | OK |
| 2026-10-28 20:55 | eval TC2-S3-ALEX-SC18-002 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.488302926 precision@20px=0.740860805; partial |
| 2026-10-28 21:13 | eval TC2-S3-ALEX-SC18-002 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.475161721 precision@20px=0.724554045; expected_count_only |
| 2026-10-28 22:00 | submit hift_s3_res18 | run TC2-S3-RES18-SC18-001 | job 48496 |
| 2026-10-28 23:18 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-29 01:12 | orchestrator v5 | resume after socket loss | tracking alex40 48453, res18 48496; res40 todo |
| 2026-10-29 01:12 | hift_s3_alex40 | 48453 TC2-S3-ALEX-SC40-001 | COMPLETED 8714s |
| 2026-10-29 01:13 | retrieve TC2-S3-ALEX-SC40-001 | final-model.pt sha256 b290dc3a1d288d36... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 01:13 | submit hift_s3_res40 | run TC2-S3-RES18-SC40-001 | job 48557 |
| 2026-10-29 01:20 | eval TC2-S3-ALEX-SC40-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.431021961 precision@20px=0.607281407; partial |
| 2026-10-29 01:34 | eval TC2-S3-ALEX-SC40-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.411629639 precision@20px=0.599990654; expected_count_only |
| 2026-10-29 01:42 | hift_s3_res18 | 48496 TC2-S3-RES18-SC18-001 | COMPLETED 8582s |
| 2026-10-29 01:42 | retrieve TC2-S3-RES18-SC18-001 | final-model.pt sha256 5a495bc29f78a06f... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 01:50 | eval TC2-S3-RES18-SC18-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.393050648 precision@20px=0.715569388; partial |
| 2026-10-29 02:06 | eval TC2-S3-RES18-SC18-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.467694347 precision@20px=0.683554018; expected_count_only |
| 2026-10-29 04:07 | hift_s3_res40 | 48557 TC2-S3-RES18-SC40-001 | COMPLETED 8682s |
| 2026-10-29 04:08 | retrieve TC2-S3-RES18-SC40-001 | final-model.pt sha256 3df05fbf5b7f031d... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 04:13 | eval TC2-S3-RES18-SC40-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.510768889 precision@20px=0.744303210; partial |
| 2026-10-29 04:25 | eval TC2-S3-RES18-SC40-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.501461794 precision@20px=0.718400103; expected_count_only |
| 2026-10-29 04:25 | orchestrator v5 | hift_b0ft_orig=retrieved hift_b0ft_corr=retrieved hift_s3_alex18=retrieved hift_s3_alex40=retrieved hift_s3_res18=retrieved hift_s3_res40=retrieved  | DONE |
| 2026-10-29 11:48 | upload | bundle c31c1a6 (sha256 22833329...), --seed + seed replicates | sha256 OK, unpacked |
| 2026-10-29 11:48 | orchestrator v6 | seed replicates s2,s3: res18/res40 + b0ft orig/corr (8 runs) | started |
| 2026-10-29 11:48 | submit hift_s3_res18_s2 | run TC2-S3-RES18-SC18-S2-001 | job 48680 |
| 2026-10-29 11:48 | diagnostic | UAV123-domain linear size ratio sqrt(pred/gt area), offline from saved predictions (GT never used by tracker): median early/late, seqs late>1.5x: RES18-SC18 1.27/1.60, 42/76; RES18-SC40 1.10/1.11, 10/76; ALEX-SC18 1.06/1.09, 15/76; ALEX-SC40 1.12/1.03, 8/76; B0FT-CORR 1.03/1.07, 6/76; B0 1.01/1.04, 19/76; earlier local RES18 1.28/2.97, 68/76 | scale 0.40 removes the ResNet size runaway (seed 1) |
| 2026-10-29 13:12 | submit hift_s3_res40_s2 | run TC2-S3-RES18-SC40-S2-001 | job 48727 |
| 2026-10-29 14:24 | hift_s3_res18_s2 | 48680 TC2-S3-RES18-SC18-S2-001 | COMPLETED 8728s |
| 2026-10-29 14:24 | retrieve TC2-S3-RES18-SC18-S2-001 | final-model.pt sha256 b36d22bf342b5e9e... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 14:38 | eval TC2-S3-RES18-SC18-S2-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.176616724 precision@20px=0.398837172; partial |
| 2026-10-29 14:59 | eval TC2-S3-RES18-SC18-S2-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.128973000 precision@20px=0.219119758; expected_count_only |
| 2026-10-29 15:35 | submit hift_b0ft_orig_s2 | run TC2-B0FT-ORIG-S2-001 | job 48768 |
| 2026-10-29 16:50 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-29 17:22 | orchestrator v6 | resume after socket loss | tracking res40_s2 48727, b0ft_orig_s2 48768; 5 todo |
| 2026-10-29 17:22 | hift_s3_res40_s2 | 48727 TC2-S3-RES18-SC40-S2-001 | COMPLETED 8649s |
| 2026-10-29 17:22 | retrieve TC2-S3-RES18-SC40-S2-001 | final-model.pt sha256 8d1d48da3b5da91a... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 17:22 | submit hift_b0ft_corr_s2 | run TC2-B0FT-CORR-S2-001 | job 48826 |
| 2026-10-29 17:34 | eval TC2-S3-RES18-SC40-S2-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.490247829 precision@20px=0.736590606; partial |
| 2026-10-29 17:51 | hift_b0ft_orig_s2 | 48768 TC2-B0FT-ORIG-S2-001 | COMPLETED 4377s |
| 2026-10-29 17:51 | retrieve TC2-B0FT-ORIG-S2-001 | final-model.pt sha256 de3f11dad25a5d2f... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 17:58 | eval TC2-S3-RES18-SC40-S2-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.510553863 precision@20px=0.716941160; expected_count_only |
| 2026-10-29 18:15 | eval TC2-B0FT-ORIG-S2-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.583396543 precision@20px=0.763471048; partial |
| 2026-10-29 18:33 | submit hift_s3_res18_s3 | run TC2-S3-RES18-SC18-S3-001 | job 48875 |
| 2026-10-29 18:35 | eval TC2-B0FT-ORIG-S2-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.582738235 precision@20px=0.784907782; expected_count_only |
| 2026-10-29 19:15 | hift_b0ft_corr_s2 | 48826 TC2-B0FT-CORR-S2-001 | COMPLETED 4385s |
| 2026-10-29 19:15 | retrieve TC2-B0FT-CORR-S2-001 | final-model.pt sha256 2a3518ea0f8213de... + summary/manifest/journal/slurm log | OK |
| 2026-10-29 19:24 | eval TC2-B0FT-CORR-S2-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.590683257 precision@20px=0.768581053; partial |
| 2026-10-29 19:41 | eval TC2-B0FT-CORR-S2-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.604905287 precision@20px=0.813014031; expected_count_only |
| 2026-10-29 20:24 | submit hift_s3_res40_s3 | run TC2-S3-RES18-SC40-S3-001 | job 48925 |
| 2026-10-29 21:20 | hift_s3_res18_s3 | 48875 TC2-S3-RES18-SC18-S3-001 | FAILED - retrying once under a new run id |
| 2026-10-29 21:20 | hift_s3_res40_s3 | 48925 TC2-S3-RES18-SC40-S3-001 | FAILED - retrying once under a new run id |
| 2026-10-29 21:20 | submit hift_s3_res18_s3 | run TC2-S3-RES18-SC18-S3-002 | job 48958 |
| 2026-10-29 21:34 | hift_s3_res18_s3 | 48958 TC2-S3-RES18-SC18-S3-002 | FAILED again - giving up on this job |
| 2026-10-29 21:34 | submit hift_s3_res40_s3 | run TC2-S3-RES18-SC40-S3-002 | job 48963 |
| 2026-10-29 21:48 | hift_s3_res40_s3 | 48963 TC2-S3-RES18-SC40-S3-002 | FAILED again - giving up on this job |
| 2026-10-29 21:48 | submit hift_b0ft_orig_s3 | run TC2-B0FT-ORIG-S3-001 | job 48967 |
| 2026-10-29 22:02 | hift_b0ft_orig_s3 | 48967 TC2-B0FT-ORIG-S3-001 | FAILED - retrying once under a new run id |
| 2026-10-29 22:02 | submit hift_b0ft_orig_s3 | run TC2-B0FT-ORIG-S3-002 | job 48972 |
| 2026-10-29 22:16 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-29 22:24 | quota | jobs 48925-48972 + res18_s3 48875 (step 17000) FAILED: OSError 122 Disk quota exceeded (home 100 GB shared; ~/hift 52 GB) | owner-approved prune: intermediate epoch ckpts of 10 retrieved runs (sha-verified locally; last full ckpt + final-model.pt kept), ckpts of 2 failed partial runs (journals kept), 4 old code copies -> ~/hift 35 GB, write OK |
| 2026-10-29 22:24 | orchestrator v6 | resume after quota fix | seed-3 runs: res18 -003, res40 -003, b0ft_orig -003, b0ft_corr -001; auto-prune after retrieval |
| 2026-10-29 22:24 | submit hift_s3_res18_s3 | run TC2-S3-RES18-SC18-S3-003 | job 49004 |
| 2026-10-29 23:48 | submit hift_s3_res40_s3 | run TC2-S3-RES18-SC40-S3-003 | job 49058 |
| 2026-10-30 00:56 | hift_s3_res18_s3 | 49004 TC2-S3-RES18-SC18-S3-003 | COMPLETED 8628s |
| 2026-10-30 00:57 | retrieve TC2-S3-RES18-SC18-S3-003 | final-model.pt sha256 7b92c89c850fe238... + summary/manifest/journal/slurm log | OK |
| 2026-10-30 01:07 | eval TC2-S3-RES18-SC18-S3-003 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.155149877 precision@20px=0.355450309; partial |
| 2026-10-30 01:24 | eval TC2-S3-RES18-SC18-S3-003 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.115269919 precision@20px=0.207606975; expected_count_only |
| 2026-10-30 02:10 | submit hift_b0ft_orig_s3 | run TC2-B0FT-ORIG-S3-003 | job 49109 |
| 2026-10-30 02:40 | orchestrator | socket check failed | STOPPED (reconnect VPN/socket) |
| 2026-10-30 03:06 | orchestrator v6 | resume after socket loss | tracking res40_s3 49058, b0ft_orig_s3 49109; b0ft_corr_s3 todo |
| 2026-10-30 03:21 | hift_s3_res40_s3 | 49058 TC2-S3-RES18-SC40-S3-003 | COMPLETED 8636s |
| 2026-10-30 03:21 | retrieve TC2-S3-RES18-SC40-S3-003 | final-model.pt sha256 aee27370890a80f8... + summary/manifest/journal/slurm log | OK |
| 2026-10-30 03:28 | eval TC2-S3-RES18-SC40-S3-003 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.433779878 precision@20px=0.646866431; partial |
| 2026-10-30 03:47 | eval TC2-S3-RES18-SC40-S3-003 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.443155450 precision@20px=0.647356873; expected_count_only |
| 2026-10-30 03:51 | submit hift_b0ft_corr_s3 | run TC2-B0FT-CORR-S3-001 | job 49128 |
| 2026-10-30 04:36 | hift_b0ft_orig_s3 | 49109 TC2-B0FT-ORIG-S3-003 | COMPLETED 4441s |
| 2026-10-30 04:36 | retrieve TC2-B0FT-ORIG-S3-003 | final-model.pt sha256 8f8746a5582bb87d... + summary/manifest/journal/slurm log | OK |
| 2026-10-30 04:46 | eval TC2-B0FT-ORIG-S3-003 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.592159001 precision@20px=0.774125870; partial |
| 2026-10-30 05:09 | eval TC2-B0FT-ORIG-S3-003 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.572217425 precision@20px=0.769147825; expected_count_only |
| 2026-10-30 05:51 | hift_b0ft_corr_s3 | 49128 TC2-B0FT-CORR-S3-001 | COMPLETED 4410s |
| 2026-10-30 05:51 | retrieve TC2-B0FT-CORR-S3-001 | final-model.pt sha256 e73b1a7dfa4a0789... + summary/manifest/journal/slurm log | OK |
| 2026-10-30 06:03 | eval TC2-B0FT-CORR-S3-001 | UAV123, sigmoid, nan-safe | sequence_macro success_auc=0.584193105 precision@20px=0.762459460; partial |
| 2026-10-30 06:26 | eval TC2-B0FT-CORR-S3-001 | VISDRONE, sigmoid, nan-safe | sequence_macro success_auc=0.597149767 precision@20px=0.805152240; expected_count_only |
| 2026-10-30 06:26 | orchestrator v6 | hift_s3_res18_s2=retrieved hift_s3_res40_s2=retrieved hift_b0ft_orig_s2=retrieved hift_b0ft_corr_s2=retrieved hift_s3_res18_s3=retrieved hift_s3_res40_s3=retrieved hift_b0ft_orig_s3=retrieved hift_b0ft_corr_s3=retrieved  | DONE |
