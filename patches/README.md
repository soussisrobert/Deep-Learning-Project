# Explicit pinned-source audit patches

Target: immutable upstream commit `992aa7c1605360d607af84cb804b93cd8ee81caf`.
These files are **not applied**. Upstream remains unchanged. Patch context deliberately preserves upstream blank-line whitespace; the parent repository exempts patch containers (not source files) from whitespace lint. Applicability is validated separately with `git apply --check`.

| Patch | Purpose | Baseline/checkpoint impact |
|---|---|---|
| [0001](<F:/Code/HiFT/patches/0001-transformer-layout-b2.patch>) | Preserve SBC→BCHW axis identity in encoder channel attention | B2 only: changes computation even at B=1; existing keys/shapes still load, but old weights are not a corrected-training baseline |
| [0002](<F:/Code/HiFT/patches/0002-negative-pair-supervision-b2.patch>) | Propagate sampled negative flag; dense background on both classifiers and zero localization weights | B2 training-policy candidate; dense rather than sampled negatives is an explicit choice, not author-intent reconstruction |
| [0003](<F:/Code/HiFT/patches/0003-reject-unsupported-distributed-launch.patch>) | Reject multi-process launch instead of pretending every rank is rank zero | Operational safety guard only; does not implement or certify DDP |
| [0004](<F:/Code/HiFT/patches/0004-singleton-class-selector-b2.patch>) | Preserve one-dimensional class selectors, use prediction device | B2 criterion edge repair; changes training only when a class subset has exactly one item; no model parameter changes |

Check without applying, from the project root:

```bash
git -C third_party/HiFT apply --check ../../patches/0001-transformer-layout-b2.patch ../../patches/0002-negative-pair-supervision-b2.patch ../../patches/0003-reject-unsupported-distributed-launch.patch ../../patches/0004-singleton-class-selector-b2.patch
```

[Patch generator](<F:/Code/HiFT/patches/build_source_patches.py>) reads upstream and writes only this
patch directory. It never applies a patch. Audit adapters and tests are separate:
[corrections](<F:/Code/HiFT/src/hift_research/corrections/tensor_layout.py>),
[tests](<F:/Code/HiFT/tests/test_source_audit.py>), [evidence report](<F:/Code/HiFT/reports/source_audit.md>).
No distributed training implementation, tiny-positive-box policy, checkpoint
migration, accuracy improvement, or completed retraining is claimed.
