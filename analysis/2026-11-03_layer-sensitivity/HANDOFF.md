# HiFT per-layer sensitivity: completed handoff

3 November2026, Singapore time. All three jobs completed successfully on TC2N01, Xeon Gold6230R. No reruns, source changes, recipe changes or TEST experiments were performed. No weights or access credentials are included. The owner must review the measured statements before updating the paper or declaring its markers cleared.

## Read first

- `results/layer_sensitivity/REPORT.md`: final report with all three tracking calibration draws.
- `fidelity.csv`, `fidelity_meta.json` within that folder:33 configuration/draw rows and identity evidence.
- `track/keep{2,3,4,5}_cal{1,2,3}/`:12 new DEV runs, each summary.json,per_seq.json and saved predictions.
- `references/results/`: original fp32 DEV reference and six round4 none/keep1 runs, with summaries and per-video scores.
- `runs/`: all3 logs, ledger, Slurm metadata and requested upstream Git check.
- `source/`: owner runbook,pre-specified plan,script,sbatch,DEV/TEST manifests and deployed helper code.
- `VALIDATION.json`, `FILE_SHA256SUMS.txt`: completeness,calibration and integrity evidence.
- `retiming/hift-stress-matched-retime-handoff.tar.gz`: completed companion study, including both timing reports,logs and identity caveats. Its one-thread trajectories do not universally match the saved four-thread accuracy run; do not call every timing identity check passed.

## Jobs

| Job | Stage | Node | State/exit | Elapsed |
|---|---|---|---|---|
| 50208 | fidelity,alone | TC2N01 | COMPLETED0:0 | 29m19s |
| 50349 | tracking,draw1 | TC2N01 | COMPLETED0:0 | 1h07m12s |
| 50350 | tracking,draws2/3 | TC2N01 | COMPLETED0:0 | 2h10m37s |

Tracking jobs ran concurrently as authorised. Their timings are descriptive only. Fidelity latency was measured alone. All jobs finished within1.5 times their estimates. VPN outages delayed result collection,not compute completion.

## Identity and completeness

All three calibration-video lists match round4. The StageA none/keep1 re-tracking check on wakeboard7 gives box-match fraction1.000 atIoU>.99. All12 new tracking runs have the exact20 DEV names and16826 frames;prediction rows and calibration names were checked locally. Calibration pools exclude both DEV and TEST. No tracebacks or explicitSTOP appeared in the job logs.

StageA used192 valid template/search pairs,not200: the unmodified ground-truth crop helper skipped8 sampled frames with invalid/absent boxes. This was independently checked against saved annotations. Twenty DEV videos were retained. Report the actual192,preserving this distinction from the plan's nominal200.

The final report contains three draws for every config. The earlier-finished job's own stdout contains a partial two-draw report because the other job was still running; do not substitute that partial table for the final REPORT.md.

## Measured answers

**Primary Q1:** layer1 is the most quantisation-sensitive by mean relative score-map L2 error when each layer is quantised alone. Ranking: layer1 .01305,layer2 .008233,layer3 .003743,layer5 .003617,layer4 .002167. This supports the narrow statement that the per-layer sweep found the largest score-map distortion in layer1. It does not by itself prove optimal tracking placement.

**Secondary Q2:** keeping layer1 is not established as the best tracking choice. All following changes are versus fp32 CPU on the same20 DEV videos,mean per video over3 draws:

| Full-precision layer | AUC change,95% paired-video interval | Excluding person19_2,as pre-specified |
|---|---|---|
| keep1 | +.0251[-.0035,+.0738] | +.0024[-.0057,+.0110] |
| keep2 | +.0464[-.0057,+.1194] | +.0250[-.0099,+.0833] |
| keep3 | +.0213[-.0051,+.0585] | +.0064[-.0081,+.0276] |
| keep4 | +.0307[-.0032,+.0793] | +.0164[-.0048,+.0543] |
| keep5 | +.0206[-.0069,+.0578] | +.0058[-.0111,+.0279] |

All these intervals overlap and include zero. Layer2 has the highest observed mean, but this small DEV study does not establish a statistically resolved best placement. Overlap alone is not a formal direct pairwise test; preserve the descriptive interpretation specified by the plan. The original chosen recipe remains unchanged. No positive accuracy-gain claim follows from these uncertain DEV means.

For context,all-int8 backbone none has+.0197[-.0102,+.0692] on all20,and-.0031[-.0129,+.0030] excludingperson19_2. That influential video matters; retain the predefined exclusion analysis without selectively hiding it. Score-map fidelity and eventual tracking behaviour answer different questions.

## Requested upstream Git check

Commit992aa7c1605360d607af84cb804b93cd8ee81caf.
Diff stat:
- pysot/models/model_builder.py:4 lines,2insertions/2deletions.
- pysot/models/utile/utile.py:4 lines,2insertions/2deletions.
- Total2 files changed,4insertions,4deletions.

Exact command output is in runs/layersens_upstream_git.txt. This records existing local upstream patches; no git modifications were made by the sensitivity task.

## Owner action

Review both this study and the companion matched re-timing report. Use measured score-map sensitivity to justify the layer1 heuristic,while describing DEV tracking placement as unresolved. Do not change the selected TEST recipe based on these DEV findings. Matched four-thread speed measurements are about1.75x network and1.72x tracker step,rather than the earlier1.80/1.76 values. One-thread measurements are about1.81x for both but carry the documented trajectory-divergence caveat. No manuscript,fact-sheet or diagram sentence has been edited. No additional experiment is authorised or queued.
