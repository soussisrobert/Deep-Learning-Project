# HiFT research engineering rules (all project files)

- Priority: correctness -> reproducibility -> diagnosis -> controlled improvement -> novelty verification.
- Upstream third_party/HiFT is an immutable submodule at 992aa7c1605360d607af84cb804b93cd8ee81caf. Never silently edit it. Keep repairs in explicit patches or isolated adapters.
- B0 = released behavior + official checkpoint, compatibility changes disclosed. B1 = independently validated inference corrections only. B2 = corrected training, controlled retraining. R1 = source-resolution candidate on frozen B2. Never merge their identities or attribute retraining gains to one fix.
- No benchmark, novelty, speed, accuracy, or checkpoint-reproduction claim without evidence. Synthetic tests establish correctness only. Label published, measured, hypothesis, and planned results distinctly.
- Source-image pixel support is not resized crop footprint. Resizing creates no new detail. No ground-truth boxes after initialization in inference controllers.
- Run CPU mathematical tests first; one GPU experiment at a time. No long training, external clusters, or large dataset downloads without explicit preparation and budget.
- Record git revision + dirty diff hash, upstream SHA, config/checkpoint hashes, data/split identity, seed, software/hardware, training recipe, metrics, runtime, limitations in immutable run manifests. Never overwrite runs.
- Validate checkpoint keys strictly; never silently use random or partially loaded weights for benchmark results. Treat external checkpoints as untrusted; prefer weights_only loading.
- Sequence-disjoint evaluation; diagnose training/pretraining leakage. UAV123 variants overlap and are not independent evidence.
- Small single-purpose modules; Python/PyTorch. Tests use pytest; deterministic invariants, malformed inputs, and numerical extremes matter.
- Parallel agents must claim shared tasks and honor explicit disjoint write scopes. Lead reviews critical mathematics and runs integrated tests.
- Do not implement R1 training until baseline/data/checkpoint validation and a preregistered R1-A diagnostic justify it. R2/R3 are deferred.
- Keep docs honest about milestone acceptance gaps and external blockers. Continue independent work when assets are missing.
