import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from automl_py import (
    AutoMLConfig,
    AutonomousAutoMLScientist,
    FeatureAvailabilityRegistry,
    SplitConformalRegressor,
)


def test_temporal_registry_blocks_future_feature():
    reg = FeatureAvailabilityRegistry({"future_actual": "+1d", "forecast": "-1d"})
    df = pd.DataFrame({"future_actual": [1, 2], "forecast": [2, 3]})
    try:
        reg.enforce(df, strict=True)
        assert False
    except ValueError:
        pass
    clean, audit = reg.enforce(df, strict=False)
    assert "future_actual" not in clean.columns
    assert audit.temporal_leakage_risk.sum() == 1


def test_conformal_regressor():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": np.linspace(0, 10, 150)})
    y = 2 * X.x + rng.normal(0, 1, 150)
    c = SplitConformalRegressor(Pipeline([("imp", SimpleImputer()), ("m", Ridge())]), alpha=0.1).fit(X, y)
    p = c.predict(X.iloc[:5])
    assert len(p.lower) == 5
    assert p.radius > 0


def test_autonomous_smoke(tmp_path):
    d = load_diabetes(as_frame=True).frame
    cfg = AutoMLConfig(
        task="regression",
        metric="r2",
        preprocessors=("original",),
        imputation_strategies=("median",),
        feature_selection=("none",),
        models=("ridge",),
        tune=False,
        cv_folds=3,
        n_jobs=1,
        stability_repeats=2,
        save_models=False,
        run_adversarial_validation=False,
        artifact_dir=str(tmp_path),
    )
    s = AutonomousAutoMLScientist(
        cfg,
        context="predict disease progression",
        feature_families={"serum": ["s1", "s2", "s3", "s4", "s5", "s6"]},
    )
    r = s.fit(d, "target")
    assert not r.automl.best_results.empty
    assert not r.experiment_plan.empty
    assert r.stability is not None
    assert r.uncertainty is not None
    assert not r.information_value.empty
    assert (tmp_path / "next_experiments.csv").exists()
