# code_ref.md

## Repository state (checked 2026-11-17)
- Git repo, remote `origin` = https://github.com/Ayushbrainer/HiFT.git, branch `main`, HEAD `992aa7c1605360d607af84cb804b93cd8ee81caf`.
- **Update (task E24, 2026-11-17): the code is committed on local branch `night-full` (not pushed), on top of BASE `992aa7c1605360d607af84cb804b93cd8ee81caf`:** commit A `f28900cf720193b168ec431bb93d6cbcedff4bec` (gamma and anchored template: `config.py`, `base_tracker.py`, `hift_tracker.py`, `tools/test.py`, `enhance.py`, `hift_tracker_tmpl.py`, `tools/test_tmpl.py`), commit B `b41f935950381e79683da78a0e0aaf028245113a` (NATDataset adapter, `eval_nat.py`, margin logging, test-set preparation), commit C `255a9fbbd9e6bcba4cb5d923bbe553d1404f3457` = branch head before this documentation commit (bank, sweep, fine-tuning, analysis, leakage and data-check scripts; job scripts as redacted copies in `jobs/` with `${HIFT_ROOT}`; the logbook as redacted `docs/experiments_log.md`). The next bullet describes the state before these commits.
- **The commit does not contain the code behind the numbers.** Untracked files (not in any commit): `pysot/utils/enhance.py`, `pysot/tracker/hift_tracker_tmpl.py`, `pysot/tracker/hift_tracker_margin.py`, `toolkit/datasets/nat.py`, `pysot/datasets/night_aug.py`, `pysot/datasets/night_dataset.py`, `pysot/utils/darken.py`, `tools/eval_nat.py`, `tools/test_tmpl.py`-family scripts and all `tools/illumination/E*.py`. Uncommitted modifications of tracked files: `pysot/core/config.py` (+9 lines, the `TRACK.ENH_*` options), `pysot/tracker/base_tracker.py` (enhance hook), `pysot/tracker/hift_tracker.py` (enhancers built in `__init__`, `enhance` passed to `get_subwindow`), `pysot/datasets/augmentation.py`, `toolkit/datasets/__init__.py`, `tools/test.py`. To make the numbers reproducible from a commit, these files must be committed; nothing was committed for this package (needs the user's decision).

## (a) Gamma
- Implementation: `pysot/utils/enhance.py` `make_enhancer` (lines 24-58): lookup table `lut = (255 * (arange(256) / 255) ** gamma).round().astype(uint8)` (line 37) applied with `cv2.LUT(patch, lut)` (line 44, inside `enhance`, mode `gamma`). Gamma 0.6. `denoise` = none and CLAHE are off in the reported runs.
- Hook: `pysot/tracker/base_tracker.py` `SiameseTracker.get_subwindow` (starts line 38): the crop is cut, padded with the frame-0 channel mean, resized with `cv2.resize` to `model_sz` (line 89), then `im_patch = enhance(np.ascontiguousarray(im_patch))` (lines 90-91), and only afterwards transposed and converted to float32 (`astype(np.float32)`, values stay 0-255).
- So gamma acts on **uint8 0-255 BGR pixels of the already resized crop, before the float conversion**. No mean / std normalisation exists in the model input path (no normalisation found by grep in `pysot/models/backbone/newalexnet.py` or `pysot/models/model_builder.py`; the network receives 0-255 floats). Crop sizes: template and template-update crops 127 x 127, search crop 287 x 287 (`cfg.TRACK.EXEMPLAR_SIZE`, `INSTANCE_SIZE`).
- Which crops: `pysot/tracker/hift_tracker.py` builds `self.enh_t` (template) and `self.enh_s` (search) from `cfg.TRACK.ENH_TEMPLATE` / `ENH_SEARCH` / `ENH_GAMMA` (lines 22-25); `init()` passes `self.enh_t` for the frame-1 template (line 104), `track()` passes `self.enh_s` for the search crop (line 139); the anchored update (below) passes `self.enh_t` too. Runs use `--enh_template gamma --enh_search gamma --gamma 0.6`.

## (b) Anchored template update (`pysot/tracker/hift_tracker_tmpl.py`, class `HiFTTemplateTracker`)
- Flags for the reported setting: `tools/test_tmpl.py --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2` (gate `none`).
- `init()` (lines 54-60): `f_hist = deque([model.zf], maxlen=3)` (the frame-1 features), `f0 = model.zf` (kept for the whole video).
- `track()` (lines 62-88): runs the plain tracker (`super().track`), then **after every tracked frame** (gate `none`: `accept` is always true) it cuts a 127 x 127 crop at the new box (`center_pos`, `size`, same context rule, same `enh_t` gamma), runs the backbone (`model.template(z)`, line 83), appends the new `model.zf` to `f_hist` (line 84), forms `avg = _mean_feat(list(f_hist))` (line 85, element-wise mean of the last 3 feature sets) and sets `model.zf = _blend(f0, avg, alpha)` (line 86), i.e. `(1 - 0.2) * f0 + 0.2 * avg` per tensor.
- Features: `model.zf` is the tuple of 3 backbone output tensors (`newalexnet.py` returns `x2, x1, x`); `_mean_feat` and `_blend` (lines 14-27) act on each of the 3 tensors separately. Because `f_hist` starts with the frame-1 features and holds the last 3 sets, the first two updates still include the frame-1 features in `avg`.
- Not implemented here: no gate in the reported setting (E8: gate adds nothing measurable), no pixel averaging.

## Scripts that produced the numbers
| Number | Script (run from `tools/`) | Job script |
|---|---|---|
| original (no enhancement) | `test.py` (or `test_margin.py` for the E13 variants) | `job_nat.sh`, `job_night_final.sh` |
| gamma 0.6 | `test.py --enh_template gamma --enh_search gamma --gamma 0.6` | `job_gamma_full.sh`, `job_nat2021l.sh`, `job_night_final.sh` |
| gamma 0.6 + anchor 0.2 | `test_tmpl.py --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2` + the gamma flags | `job_p15.sh`, `job_anchor180.sh`, `job_anchorL.sh`, `job_night_final.sh` |
| official-code per-video CSVs | `tools/illumination/E23_package.py` (copy: `figures/E23_package.py`) | `job_E23.sh` |
| mean success / precision tables | `tools/eval_nat.py` | same jobs |
| bootstraps (E16, E19) | `tools/ablation_boot.py`, `tools/illumination/E19_settings_bootstrap.py` | `job_ablation_boot.sh`, `job_E19.sh` |
| lost frames / stretches (E15) | `tools/lost_diag.py` | `job_lost_diag.sh` |
| template similarity (E17) | `tools/template_sim.py`, `template_sim_analysis.py` | `job_template_sim.sh` |
| oracle bank (E18) | `pysot/tracker/hift_tracker_bank.py`, `tools/test_bank.py`, `tools/illumination/E18_analysis.py` | `job_bank.sh` |
| size oracle (E20), E21 sweep, E22 taxonomy | `tools/illumination/E20_size_oracle.py`, `E21_*.py` + `tools/test_tmpl_sweep.py`, `E22_*.py` | `job_E20.sh`, `job_E21.sh`, `job_E22.sh`, `job_E22D.sh` |
| fine-tuning (A-D) | `tools/train_night.py`, `pysot/datasets/night_dataset.py`, `night_aug.py`, configs `experiments/night_ft/run_*.yaml` | `job_night_train.sh`, `job_night_final.sh` |
