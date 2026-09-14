from __future__ import annotations
from contextlib import contextmanager


@contextmanager
def mlflow_run(enabled: bool, experiment: str, run_name: str = "automl-scientist"):
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("Install automl-py[tracking] to enable MLflow tracking.") from exc
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield mlflow
