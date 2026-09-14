# Contributing

1. `pip install -e ".[dev]"`
2. `pytest -q`
3. `ruff check .` and `ruff format --check .`
4. Keep learned transformations inside sklearn-compatible fit/transform pipelines.
5. Never compute target-aware or dataset-wide learned statistics before the split.
6. Every experiment must run on the shared `FoldSet`; never build ad-hoc splits inside an experiment.
7. Never read the `FinalHoldout` for selection, planning, or tuning. It is evaluated once, for the champion.
8. A candidate is only `KEEP` through `ExperimentJudge` (paired folds, noise floor, minimum gain, confidence interval).
9. Do not swallow exceptions: record a `DiagnosticEvent`.
10. Documentation must label capabilities IMPLEMENTED / EXPERIMENTAL / PLANNED; do not claim more than the tests demonstrate.
