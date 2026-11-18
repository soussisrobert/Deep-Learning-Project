# Matched HiFT CPU re-timing handoff

2 November2026. Both authorised jobs completed successfully on TC2N01,Intel Xeon Gold6230R,torch2.8.0+cu128. Four-thread job50072 elapsed1:31:46;one-thread job50143 elapsed1:48:31. No other jobs of this account ran alongside them. Exact owner commands and settings were retained; no weights or credentials were copied.

## Results to review before updating the paper

Four-thread six-session matched measurements: median network speedup1.7484x,range1.7384–1.7600x; median tracker-step speedup1.7170x,range1.7113–1.7308x. One-thread four-session measurements: median network1.8134x,range1.8048–1.8174x; median tracker-step1.8054x,range1.7917–1.8339x. These are medians/ranges of within-session ratios, with alternating model order. Tracker-step excludes image loading. Four fixed TEST timing videos,bike1,car1_2,person11,person7_2,5496 tracked frames per model per session. Sizes44.3MB fp32,33.2MB int8. Do not report these as full-system/drone speedups.

Calibration names matched the accuracy-tested run in both jobs,with158cropsmatching. Four-thread saved-box identity shares were1.000 on every checked video for fp32 andint8 in both checked sessions. One-thread box identity DID NOT fully match the saved four-thread accuracy run. AtIoU>.99,int8 shares for bike1/car1_2/person11/person7_2 were0.931/1.000/0.576/0.368 in both checked sessions;fp32 shares0.470/1.000/1.000/1.000. Exact unrounded metrics,max pixel errors,per-video AUC andmedianIoUs are in summary.json. Same CPU model in both jobs;do not attribute divergence to a different CPU without evidence. Report divergence as required by the runbook,without retries or changes. Calibration identity passed;one-thread trajectory identity did not universally pass. Do not call all identity checks passed or claim one-thread accuracy identical to the saved four-thread experiment.

## Included

results/retime_matched_t4 and_t1 each contain REPORT.md,summary.json,sessions.csv. runs/ contains both full logs,retime_jobs.tsv and Slurm metadata. source/ contains the exact script,sbatch andrunbook;SOURCE_SHA256SUMS.txt records their hashes. EXECUTION.md is the operational record. All session counts,alternating orders,ratio median/min/max andCPU/calibration fields were locally verified. No traceback or explicitSTOP was found. Scripts and paper were not edited.

## Next owner action

Review these measurements and the one-thread box divergence,then update approved paper/fact-sheet/diagram sentences. The old1.80x four-thread network claim should not be repeated as the matched result;the replacement is about1.75x network and1.72x tracker-step,subject to owner review. Layer-sensitivity work is separately authorised and running after timing completion;its package will follow. No inference about per-layer sensitivity is made here.
