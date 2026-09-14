from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .validation import ValidationConfig, default_validation

Task = Literal["regression", "classification"]
Metric = Literal["r2", "rmse", "mae", "accuracy", "roc_auc", "f1"]
Preprocess = Literal["original", "scale", "pca"]
ImputationStrategy = Literal["median", "mean", "most_frequent", "knn", "iterative"]


@dataclass(slots=True)
class AutoMLConfig:
    """Configuration surface. Every field here is consumed by the implementation.

    Fields marked EXPERIMENTAL are wired but lightly exercised; nothing here is
    decorative scaffolding.
    """

    task: Task = "regression"
    metric: Metric | None = None
    preprocessors: Sequence[Preprocess] = ("original", "scale", "pca")
    models: Sequence[str] | None = None
    imputation_strategies: Sequence[ImputationStrategy] = ("median",)
    add_missing_indicators: bool = True
    interaction_terms: bool = False
    interaction_degree: int = 2
    feature_selection: Sequence[str] = ("none",)
    feature_selection_k: int | str = "all"

    # Validation design -------------------------------------------------------
    # ``validation`` is the first-class validation strategy shared by every
    # experiment. When None, a seeded random (regression) or stratified
    # (classification) K-fold is built from cv_folds / cv_repeats.
    validation: ValidationConfig | None = None
    # Fraction of rows (or, for temporal strategies, of periods) reserved as the
    # locked final holdout. It is evaluated exactly once, for the champion.
    test_size: float = 0.20
    holdout_periods: int | None = None
    # When False the holdout stays locked after model search (the orchestrator
    # evaluates it once after the feature set is final).
    evaluate_holdout: bool = True
    random_state: int = 100
    cv_folds: int = 5
    cv_repeats: int = 1
    n_jobs: int = -1
    pca_variance: float = 0.95
    tune: bool = True
    max_candidates_per_model: int = 16

    # Scientist diagnostics -----------------------------------------------------
    run_data_profile: bool = True
    run_missingness_analysis: bool = True
    run_leakage_detection: bool = True
    run_residual_analysis: bool = True
    run_adversarial_validation: bool = True
    run_feature_ablation: bool = False  # EXPERIMENTAL: champion ablation on shared folds (cost = one CV run per feature)
    run_stability_analysis: bool = True
    run_noise_controls: bool = True
    run_uncertainty: bool = True

    # Experiment judgement -----------------------------------------------------
    # Practical-significance floor applied by ExperimentJudge to every paired
    # experiment (feature families, candidate datasets, discovery loop).
    minimum_feature_gain: float = 0.005  # minimum absolute gain in metric units
    minimum_relative_gain: float = 0.0  # minimum gain as a fraction of |baseline|
    noise_controls: int = 5  # number of random/permuted control features when run_noise_controls
    noise_quantile: float = 0.95
    uplift_ci_level: float = 0.95
    min_positive_fold_share: float = 0.6

    # Autonomous loop -----------------------------------------------------------
    autonomous_rounds: int = 1  # maximum orchestrator iterations when a candidate source is supplied
    max_experiments: int = 50
    no_improvement_rounds: int = 3
    stability_repeats: int = 8
    uncertainty_alpha: float = 0.10
    max_ablation_features: int = 30
    temporal_strict: bool = True
    unknown_availability_policy: Literal["review", "strict", "allow"] = "review"

    save_models: bool = True
    artifact_dir: str = "automl_artifacts"
    enable_mlflow: bool = False
    mlflow_experiment: str = "AutoML Scientist"

    extra: dict = field(default_factory=dict)

    def resolved_metric(self) -> Metric:
        return self.metric or ("accuracy" if self.task == "classification" else "r2")

    def resolved_validation(self) -> ValidationConfig:
        if self.validation is not None:
            return self.validation.validate()
        return default_validation(self.task, self.cv_folds, self.cv_repeats, self.random_state)
