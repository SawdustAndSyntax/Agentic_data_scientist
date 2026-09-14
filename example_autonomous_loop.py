"""The iterative Predictive Discovery loop on a synthetic store x week demand problem."""

import numpy as np
import pandas as pd

from automl_py import (
    AutoMLConfig,
    AutonomousAutoMLScientist,
    CandidateFeatureSet,
    InMemoryCandidateSource,
    StopConfig,
    ValidationConfig,
)

rng = np.random.default_rng(7)
n_weeks, n_stores = 120, 4
weeks = np.repeat(np.arange(1, n_weeks + 1), n_stores)
stores = np.tile(np.arange(1, n_stores + 1), n_weeks)
n = len(weeks)
temperature = 20 + 10 * np.sin(2 * np.pi * weeks / 52) + rng.normal(0, 2, n)
promo = rng.binomial(1, 0.3, n)
x = rng.normal(size=n)
sales = 50 + 3 * x + 2.5 * temperature + 15 * promo + rng.normal(0, 2, n)
df = pd.DataFrame({"week": weeks, "store_id": stores, "x": x, "sales": sales})

# Candidate frames are index-aligned with df. Availability is declared per column.
source = InMemoryCandidateSource(
    [
        CandidateFeatureSet(
            "weather_daily",
            pd.DataFrame({"temperature": temperature}, index=df.index),
            ("weather", "temperature"),
            availability={"temperature": "available"},
            cost=85_000,
        ),
        CandidateFeatureSet(
            "promotions", pd.DataFrame({"promo": promo}, index=df.index), ("price_promotion",), availability={"promo": "available"}
        ),
        CandidateFeatureSet(
            "fuel_prices",
            pd.DataFrame({"fuel": rng.normal(size=n)}, index=df.index),
            ("economy", "fuel"),
            availability={"fuel": "available"},
        ),
        CandidateFeatureSet(
            "final_invoice",
            pd.DataFrame({"final_invoice": sales * 1.01}, index=df.index),
            ("events",),
            availability={"final_invoice": "unavailable"},
        ),
    ]
)

config = AutoMLConfig(
    task="regression",
    metric="rmse",
    preprocessors=("original",),
    models=("ridge",),
    tune=False,
    n_jobs=1,
    save_models=False,
    validation=ValidationConfig(
        strategy="rolling_origin", timestamp_column="week", group_columns=["store_id"], min_train_periods=40, test_periods=10, step=10
    ),
    minimum_feature_gain=0.5,
    noise_controls=4,
    autonomous_rounds=5,
)

scientist = AutonomousAutoMLScientist(
    config,
    context="weekly store ice cream sales demand",
    candidate_source=source,
    stop=StopConfig(max_iterations=5, no_improvement_rounds=2),
)
result = scientist.fit(df, "sales")
loop = result.loop
print(loop.summary())
print(loop.audit_trail[["experiment_id", "iteration", "candidate", "decision", "mean_uplift", "confidence_interval"]])
print(loop.model_card)
