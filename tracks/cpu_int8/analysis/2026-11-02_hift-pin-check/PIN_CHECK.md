# HiFT code pin and local changes: verification (2 Nov 2026)

Read-only check: nothing was pushed, committed or changed.

The checkout examined here was built in Claude's workspace by the repository's own setup script (`setup_local.sh`). That is the same clone → checkout → patch sequence as `slurm/job_setup_tc2.sbatch` uses on TC2. The cluster checkout should still be confirmed with the command in §6.

## 1. How HiFT is included
- **Not a submodule.** There is no `.gitmodules`, and `.gitignore` line 3 excludes `third_party/`. So `third_party/HiFT` is **not in our GitHub repo at all**.
- **It is cloned and pinned by the setup scripts:**
  - `setup_local.sh`: line 8 `PIN=992aa7c1605360d607af84cb804b93cd8ee81caf`; lines 23–24 `git clone https://github.com/vision4robotics/HiFT.git third_party/HiFT` and `git -C third_party/HiFT checkout "$PIN"`; line 26 `python tools/patch_hift.py third_party/HiFT`.
  - `slurm/job_setup_tc2.sbatch`: the same, at lines 13, 29–33.
  - `tools/patch_hift.py` also hard-codes the pin (line 13) and refuses to patch if the expected lines are not found exactly once.

## 2–3. The commit and upstream
- **Full SHA:** `992aa7c1605360d607af84cb804b93cd8ee81caf` (starts with `992aa7c` ✓).
- **Upstream repository:** `https://github.com/vision4robotics/HiFT`. **The HiFT paper links to this repository**: its abstract ends "Our code is available at https://github.com/vision4robotics/HiFT." (from the PDF text, `05_literature/pdfs/cao2021hift.pdf`).
- **`git ls-remote https://github.com/vision4robotics/HiFT.git` (verbatim, first lines):**
  ```
  992aa7c1605360d607af84cb804b93cd8ee81caf	HEAD
  992aa7c1605360d607af84cb804b93cd8ee81caf	refs/heads/main
  ```
  So the pinned commit is the **current head of upstream `main`**. `git branch -r --contains 992aa7c` gives `origin/HEAD -> origin/main`, `origin/main`.
- **Commit date and message:**
  ```
  992aa7c1605360d607af84cb804b93cd8ee81caf
   author date: 2022-10-11 16:11:55 +0800
   commit date: 2022-10-11 16:11:55 +0800
   message: update
  ```

## 4. Every change against the commit

**a. Edits inside the checkout (source files).** `git diff --stat -- '*.py' '*.yaml'`:
```
 pysot/models/model_builder.py | 4 ++--
 pysot/models/utile/utile.py   | 4 ++--
 2 files changed, 4 insertions(+), 4 deletions(-)
```

| File | Line | Before | After | Device-only? |
|---|---|---|---|---|
| `pysot/models/model_builder.py` | 22 | `self.backbone = AlexNet().cuda()` | `self.backbone = AlexNet()` | Yes |
| `pysot/models/model_builder.py` | 23 | `self.grader=HiFT(cfg).cuda()` | `self.grader=HiFT(cfg)` | Yes |
| `pysot/models/utile/utile.py` | 96 | `i = t.arange(w).cuda()` | `i = t.arange(w, device=res3.device)` | Yes |
| `pysot/models/utile/utile.py` | 97 | `j = t.arange(h).cuda()` | `j = t.arange(h, device=res3.device)` | Yes |

All four are made by `tools/patch_hift.py`. No other source line differs.

**b. Compiled-cache files.**
- The upstream commit also tracks **82 compiled `.pyc` cache files** (Python 3.7/3.8 `__pycache__`). In this workspace checkout they show as deleted.
- They are bytecode caches, not source. Python 3.11, which all our runs use, never loads `cpython-37`/`cpython-38` caches.
- So they cannot affect results. The full `git diff --stat` therefore reads "84 files changed, 4 insertions(+), 4 deletions(-)": the 2 source files plus 82 cache files.
- §6 shows whether the cluster checkout has them.

**c. Runtime changes made by our adapter (no file in HiFT is touched):**

| Where | What | Effect | Device-only? |
|---|---|---|---|
| `hiftstress/tracker.py:19` | `cfg.CUDA = False` | HiFT's `base_tracker.py:93` then skips its own `.cuda()` on each crop | Yes |
| `hiftstress/tracker.py:112–113` | `HookedTracker.get_subwindow` = HiFT's `get_subwindow(...)` followed by `.to(self.device)` | Moves the crop to the chosen device | Yes |
| `hiftstress/tracker.py`, `TrackingModel` | Wraps the network call with optional attack / defence / corruption hooks; logs confidence; casts outputs to float | Experiment conditions. In the clean fp32 runs all hooks are off, and the cast is a no-op | Not HiFT changes (experiment wrappers) |
| `hiftstress/model.py:39–50` | `safe_load`: `torch.load(..., weights_only=True)` with numpy scalar types allow-listed | Safer loading of the same checkpoint. Summaries record `missing=[]`, `unexpected=[]` | Not a model change |
| `hiftstress/model.py:57` | `torch.manual_seed(0)` before building | Only affects initial random weights, which the checkpoint overwrites | Not a model change |
| `hiftstress/edge.py` | Quantisation and pruning variants build modified *copies* of the network (e.g. `setattr` on the head only for int8full) | Our experimental variants. The fp32 baseline is untouched | Not HiFT changes (our method) |

## 5. Configuration
- **Our runs load** `third_party/HiFT/experiments/config.yaml`, merged into HiFT's own `pysot/core/config.py` defaults (`hiftstress/model.py:17, 24`).
- **`git diff --exit-code -- experiments/config.yaml pysot/core/config.py` → no differences.** Both are identical to the pinned commit.
- The only runtime override is `cfg.CUDA = False` (§4c).

## 6. Cluster checkout
- The workspace checkout is built by the same scripts, but **the results were produced from the TC2 checkout**, which GitHub cannot show (`third_party/` is git-ignored).
- **The cluster command is still needed** to confirm the TC2 copy has the same HEAD and only these 4 source edits:
  ```bash
  git -C third_party/HiFT rev-parse HEAD && git -C third_party/HiFT diff --stat && git -C third_party/HiFT diff
  ```
- The layer-sensitivity agent was asked to run it.
- **Expected:** HEAD `992aa7c1605360d607af84cb804b93cd8ee81caf`; the same 2 source files and 4 lines, plus possibly the `.pyc` deletions.

## Suggested sentence (once the cluster output matches)

> "Our checkout is the official commit `992aa7c1605360d607af84cb804b93cd8ee81caf` (vision4robotics/HiFT, the repository linked from the HiFT paper), changed only in 4 lines across 2 files to replace hard-coded CUDA calls with device-neutral equivalents; the configuration files are unchanged."

## Cluster confirmation (3 Nov)

The TC2 checkout that produced the results gives, verbatim:
```
992aa7c1605360d607af84cb804b93cd8ee81caf
 pysot/models/model_builder.py | 4 ++--
 pysot/models/utile/utile.py   | 4 ++--
 2 files changed, 4 insertions(+), 4 deletions(-)
```
- **Match:** the same HEAD, and the same 2 source files and 4 lines. No `.pyc` deletions on the cluster.
- **The marker can be replaced** with the suggested sentence above.
- **Source:** `../2026-11-03_layer-sensitivity/cluster_hift_git_check.txt`.

