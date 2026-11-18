# Data

Datasets are not redistributed. Download them from their publishers; the exact sequence lists we used
are in `splits/`.

| Dataset | Used by | Sequences used | Rate / stride | Source |
|---|---|---|---|---|
| UAV123 | CPU | all 123 (DEV-20, TEST-40, non-TEST-83 lists in `splits/cpu/`) | 30 fps, every frame | UAV123 benchmark (Mueller et al., ECCV 2016) |
| UAV123@10fps | domain | 76 car/person/truck/group segments (`splits/domain/uav123_10fps_76_selection.json`) | 10 fps release, every frame | distributed with UAV123 |
| NAT2021 test | night | 180 (`splits/night/NAT2021_180.txt`) | released frames, every frame | NAT2021 (Ye et al., CVPR 2022, UDAT) |
| NAT2021-L test | night | 23 (`splits/night/NAT2021L_23.txt`) | released frames, every frame | NAT2021 (Ye et al., CVPR 2022, UDAT) |
| UAVDark135 | night | test 23 / val 16 (train 96 used only for unreported night fine-tuning); linked-footage groups in `splits/night/UAVDark135_groups.csv` | released frames, every frame | UAVDark135 (Li et al. 2021, All-Day Object Tracking for UAV) |
| VisDrone2019-SOT test-dev | domain | 34 of 35 (source-disjoint from training; `splits/domain/source_cluster_mapping.csv`) | released frames, every frame | VisDrone (Zhu et al. 2018); our copy came from a Hugging Face mirror whose archive structure matches the official release |
| VisDrone MOT/SOT train, UAVDT | domain training | see `tracks/domain_ft/evidence/stage7/docs/datasets.md` | — | VisDrone; UAVDT |

Frame-level content hashes for the domain evaluation sets are in
`tracks/domain_ft/evidence/stage7/artifacts/domain-*/dataset_manifest.json`.
Expected UAV123 size: 123 sequences / 112,578 frames (`tracks/cpu_int8/plan/uav123_frames.json`).
