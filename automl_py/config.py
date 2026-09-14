from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence

Task = Literal["regression", "classification"]
Metric = Literal["r2", "rmse", "mae", "accuracy", "roc_auc", "f1"]
Preprocess = Literal["original", "scale", "pca"]
ImputationStrategy = Literal["median", "mean", "most_frequent", "knn", "iterative"]


@dataclass(slots=True)
class AutoMLConfig:
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
    test_size: float = 0.20
    random_state: int = 100
    cv_folds: int = 5
    cv_repeats: int = 1
    n_jobs: int = -1
    pca_variance: float = 0.95
    tune: bool = True
    max_candidates_per_model: int = 16

    # Scientist diagnostics
    run_data_profile: bool = True
    run_missingness_analysis: bool = True
    run_leakage_detection: bool = True
    run_residual_analysis: bool = True
    run_adversarial_validation: bool = True
    run_feature_ablation: bool = True
    run_stability_analysis: bool = True
    run_noise_controls: bool = True
    run_uncertainty: bool = True

    # v0.6 autonomous behavior
    autonomous_rounds: int = 1
    stability_repeats: int = 8
    uncertainty_alpha: float = 0.10
    max_ablation_features: int = 30
    minimum_feature_gain: float = 0.005
    temporal_strict: bool = True

    save_models: bool = True
    artifact_dir: str = "automl_artifacts"
    enable_mlflow: bool = False
    mlflow_experiment: str = "AutoML Scientist"

    def resolved_metric(self) -> Metric:
        return self.metric or ("accuracy" if self.task == "classification" else "r2")
