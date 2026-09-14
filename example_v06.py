"""Judged feature-family experiments on shared folds with a locked holdout."""

from sklearn.datasets import load_diabetes

from automl_py import AutoMLConfig, AutonomousAutoMLScientist, FeatureAvailabilityRegistry

frame = load_diabetes(as_frame=True).frame

config = AutoMLConfig(
    task="regression",
    metric="rmse",
    preprocessors=("original", "scale"),
    imputation_strategies=("median",),
    feature_selection=("none",),
    models=("ridge", "hist_gb"),
    tune=True,
    n_jobs=-1,
    stability_repeats=6,
    minimum_feature_gain=0.5,  # practical floor in RMSE units
    noise_controls=5,
)

# Declare when features are knowable. Undeclared features are UNKNOWN and flagged for review.
availability = FeatureAvailabilityRegistry({c: "0s" for c in frame.columns if c != "target"})

scientist = AutonomousAutoMLScientist(
    config,
    context="predict disease progression from available measurements",
    feature_availability=availability,
    feature_families={
        "body_composition": ["bmi"],
        "blood_pressure": ["bp"],
        "serum_markers": ["s1", "s2", "s3", "s4", "s5", "s6"],
    },
)

result = scientist.fit(frame, "target")
print(result.summary())
print(result.information_value[["feature_family", "decision", "mean_uplift", "ci_low", "ci_high", "positive_share", "noise_threshold"]])
print(result.next_experiments())
