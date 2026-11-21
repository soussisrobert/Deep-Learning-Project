Robert's HiFT corner refinement handoff
Prepared 21 November 2026

The primary CSV

per_video_VisDrone34.csv contains 68 rows, paired over the exact 34 domain test videos and 22 source clusters. It reports AUC and precision@20 from the domain evaluate_sequence implementation, SHA-256 53216b126ae8dcc57cd9831bb24226bb80e1f21ea7c80f256026086a55c747f2. Initialization is included. Scores are fractions from 0 to 1. Frames means every scored row including initialization.

This is a fresh replay using the existing trained weights. Stock AUC is 0.529978597, final AUC is 0.545623692, and the paired gain is 0.015645095. The 95% source-cluster interval is [0.005268586, 0.027976986]. It uses the common repository's shared/stats.py, 10,000 paired cluster resamples, seed 20261105. Reported intervals measure source variation for one trained seed. Earlier test-set exposure makes this an exploratory result.

The original report and the baseline difference

per_video_historical_VisDrone35.csv exports the untouched original TC2 per-video AUC values. Initialization was excluded. Precision@20 is blank because it was not recorded. Its 35-clip means were 0.534605518 and 0.550660882. Removing uav0000079_02568_s gives 0.529316483 and 0.544884421. This is a subset calculation from the saved results.

The domain B0 baseline is 0.526709200 on 34 clips. Its sealed inference manifest uses sigmoid scores and NaN-safe selection on an RTX 5070 Ti GPU. The recorded NaN-safe guard had zero affected frames. Our baseline keeps the released raw cls2 score mixing and runs on CPU. Both use official weights with SHA-256 8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101. The dataset video names and all ground-truth arrays match the domain manifest. The strict domain CLI also verified the original annotation and image hashes on all 34 retained clips. The full CLI evaluations are included under evidence/.

The domain scorer includes initialization. Applying only this scoring adjustment to the original 34-clip AUC values would give 0.530025306 and 0.545601202. The inference differences still apply. The primary CSV uses actual saved replay boxes and the domain scorer.

The implemented pipeline

Frozen stock HiFT produces a box. The trained coarse refiner supplies a proposal. The corner head compares a first-frame target crop to a fresh crop spanning 2.5 times the proposal width and height, using frozen shallow HiFT features and learned template matching. It predicts top-left and bottom-right corner heatmaps and visibility. At visibility >= 0.65, the final box moves 35% from the coarse proposal toward the corner prediction. Otherwise the coarse proposal is reported. These corrections affect the output box. HiFT's own internal state controls the next frame's search.

The full historical 34-clip gain over stock is +1.5568 AUC points. The corner stage's incremental gain over the coarse refiner is +0.5601 points. Validation chose 35% among the corner settings, although the coarse refiner alone had higher validation AUC than the complete pipeline. The earlier refiner had already been evaluated on the same test clips. The quick PDF's +1.61 points refers to the full 35-clip pipeline.

Training provenance

The corner head used 86 SOT training clips / 66 source IDs / 69,941 frames. Its warmup used 7,994 jittered crops for 18 epochs. Fine-tuning used 10,830 actual tracker proposals for 12 epochs, on the same training clips. The coarse head used 43,281 accepted proposals from that training split. The official backbone was frozen. AdamW, batch 32, seed 0, cosine learning-rate schedule. Corner warmup learning rate 0.001, fine-tuning learning rate 0.0004, schedule minimum 0.0001. Training and fine-tuning video lists are identical and included in manifests/train_videos.txt.

Validation used the 11 official SOT validation clips / 9 source IDs / 7,046 frames. The complete training, validation and test manifests share no source ID prefixes. This is a filename-based source-flight audit of the added head training. The official checkpoint's pretraining membership remains unknown.

The original TC2 environment used Python 3.11 and torch 2.8.0+cpu, with OpenCV 4.12.0. The replay's software versions and per-video differences from the original run are in audit.json and evidence/replay_per_video.json. Replays on another library version or platform may differ.

Code and checkpoints

Local branch robert-corner-refinement-handoff
Local snapshot commit 62167ca7728838543f6344d3625dcc5f1f5bb9a8
Pinned upstream commit 992aa7c1605360d607af84cb804b93cd8ee81caf
Team scorer repository commit f850462a54c731b9c7c2cb89c2cf3a36fd63f5d7

The original experiment code had uncommitted additions. The bundle records a local reproducibility snapshot created on 21 November, plus the export runner used for this replay. The bundle can be imported into a checkout of the official upstream repository. Its dependencies include helper modules used by earlier experiments because the original trainer imports those modules.

checkpoints/corner-head.pt is the final trained corner head.
checkpoints/coarse-refiner.pt is the required coarse proposal head.
checkpoints/warmup-corner-head.pt is the initial corner head used for fine-tuning.
All checkpoint hashes are in audit.json and SHA256SUMS. The 86 MB official HiFT checkpoint is available through the official upstream README and is already present in Robert's local checkout.

Restore the source with Git

git clone https://github.com/vision4robotics/HiFT.git hift-corner
cd hift-corner
git fetch /absolute/path/to/handoff/source/corner-refinement.bundle HEAD:robert-corner-refinement-handoff
git switch robert-corner-refinement-handoff

Use config/experiments_config.yaml from this package byte-for-byte when calling the replay runner. The training checkpoint hashes the config file, including its line endings. The included config preserves that identity.

Run the replay from the restored HiFT repository

python -m tools.export_corner_results --manifest HANDOFF/manifests/disjoint34.json --frames FRAME_ROOT --snapshot OFFICIAL_CHECKPOINT --config HANDOFF/config/experiments_config.yaml --corner-head HANDOFF/checkpoints/corner-head.pt --version-one HANDOFF/checkpoints/coarse-refiner.pt --domain-metrics HANDOFF/scoring/domain_metrics.py --output NEW_OUTPUT_DIRECTORY --threads 1

Replace HANDOFF, FRAME_ROOT and OFFICIAL_CHECKPOINT with actual paths. FRAME_ROOT must contain the archive-prefixed paths recorded in the manifest. The replay requires PyTorch, NumPy, OpenCV, PyYAML and yacs. It writes every model's boxes and per-video score records. Training can be reproduced with tools/train_corner_refiner.py and the included SLURM recipe, with paths adjusted to the supplied checkpoints and manifests.

Verify this package without inference

python verify_handoff.py

This needs NumPy. It verifies every listed file hash, all 68 primary score rows against the saved boxes, the exact domain test membership, source separation across splits and the common paired cluster interval. comparison/domain_B0_REF_VisDrone34.csv is copied from the team repository to make the baseline distinction explicit. It represents a separate inference experiment.
