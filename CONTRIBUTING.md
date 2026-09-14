# Contributing

1. `pip install -e ".[dev]"`
2. `pytest -q`
3. `ruff check .`
4. Keep learned transformations inside sklearn-compatible fit/transform pipelines.
5. Never compute target-aware or dataset-wide learned statistics before the split.
6. For temporal problems, honor information availability and use chronological validation.
