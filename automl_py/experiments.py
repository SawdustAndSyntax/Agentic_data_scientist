"""Paired experiments on identical folds, with noise controls and a judge.

Every candidate is compared with its baseline fold-by-fold on the same
FoldSet. Random and permuted control features establish an empirical noise
floor. The ExperimentJudge turns the paired differences into a decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from .config import AutoMLConfig
from .judge import ExperimentJudge, ExperimentVerdict, paired_uplift
from .metrics import higher_is_better, scorer_name
from .validation import FoldSet


@dataclass(frozen=True)
class FeatureExperiment:
    name: str
    columns: tuple[str, ...]
    rationale: str = ""


@dataclass
class ExperimentResult:
    """A completed paired experiment and its verdict."""

    name: str
    metric: str
    higher_is_better: bool
    baseline_columns: list[str]
    candidate_columns: list[str]
    baseline_scores: list[float]
    candidate_scores: list[float]
    verdict: ExperimentVerdict
    validation_strategy: str
    noise_uplifts: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def decision(self) -> str:
        return self.verdict.decision

    @property
    def established(self) -> bool:
        return self.verdict.established

    @property
    def baseline_mean(self) -> float:
        return self.verdict.baseline_mean

    @property
    def candidate_mean(self) -> float:
        return self.verdict.candidate_mean

    def to_dict(self) -> dict:
        d = {
            "experiment": self.name,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "validation_strategy": self.validation_strategy,
            "baseline_columns": list(self.baseline_columns),
            "candidate_columns": list(self.candidate_columns),
            "baseline_scores": list(self.baseline_scores),
            "candidate_scores": list(self.candidate_scores),
            "noise_uplifts": list(self.noise_uplifts),
        }
        d.update(self.verdict.to_dict())
        d.update(self.extra)
        return d

    def to_row(self) -> dict:
        d = self.to_dict()
        for k in (
            "baseline_columns",
            "candidate_columns",
            "baseline_scores",
            "candidate_scores",
            "noise_uplifts",
            "fold_uplifts",
            "reasons",
            "invalid_reasons",
            "review_reasons",
        ):
            if k in d and isinstance(d[k], list):
                d[k] = ", ".join(str(x) for x in d[k]) if k.endswith("columns") or k.endswith("reasons") else str(d[k])
        return d


def judge_from_config(config: AutoMLConfig) -> ExperimentJudge:
    return ExperimentJudge(
        minimum_absolute_gain=config.minimum_feature_gain,
        minimum_relative_gain=config.minimum_relative_gain,
        noise_quantile=config.noise_quantile,
        ci_level=config.uplift_ci_level,
        ci_method=config.uplift_ci_method,
        min_positive_share=config.min_positive_fold_share,
        random_state=config.random_state,
    )


def ensure_foldset(cv, X, y=None) -> FoldSet:
    """Materialize any sklearn splitter into a FoldSet so all experiments share rows."""
    if isinstance(cv, FoldSet):
        if len(X) != cv.n_rows:
            raise ValueError(f"FoldSet has {cv.n_rows} rows but X has {len(X)}")
        return cv
    folds = [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(X, y)]
    return FoldSet(folds, type(cv).__name__, len(X), X.index if hasattr(X, "index") else None)


def _fresh(model_or_factory, X):
    return model_or_factory(X) if callable(model_or_factory) and not hasattr(model_or_factory, "fit") else clone(model_or_factory)


def fold_scores(model_or_factory, X: pd.DataFrame, y, folds: FoldSet, metric: str, n_jobs: int = 1) -> list[float]:
    """Per-fold scores in metric units (RMSE/MAE positive, lower is better)."""
    raw = cross_val_score(_fresh(model_or_factory, X), X, y, cv=folds, scoring=scorer_name(metric), n_jobs=n_jobs, error_score=np.nan)
    return [(-float(s) if metric in {"rmse", "mae"} else float(s)) for s in raw]


def noise_control_uplifts(
    model_or_factory,
    X: pd.DataFrame,
    y,
    folds: FoldSet,
    metric: str,
    base_cols: Sequence[str],
    baseline_scores: Sequence[float],
    *,
    n_controls: int = 5,
    random_state: int = 100,
    n_jobs: int = 1,
) -> list[float]:
    """Mean paired uplift of each random / permuted control feature added to the baseline.

    Controls alternate between Gaussian noise and a permuted copy of a real
    baseline column so the floor reflects what 'a useless feature' looks like
    for this model on these folds.
    """
    rng = np.random.default_rng(random_state)
    base_cols = list(base_cols)
    numeric = [c for c in base_cols if pd.api.types.is_numeric_dtype(X[c])]
    out = []
    for i in range(int(n_controls)):
        Xn = X[base_cols].copy()
        if i % 2 == 1 and numeric:
            src = numeric[rng.integers(len(numeric))]
            Xn[f"__noise_permuted_{i}"] = rng.permutation(X[src].to_numpy())
        else:
            Xn[f"__noise_normal_{i}"] = rng.normal(size=len(Xn))
        scores = fold_scores(model_or_factory, Xn, y, folds, metric, n_jobs)
        out.append(paired_uplift(baseline_scores, scores, higher_is_better=higher_is_better(metric), n_bootstrap=1).mean)
    return out


def run_paired_experiment(
    name: str,
    model_or_factory,
    X: pd.DataFrame,
    y,
    folds: FoldSet,
    metric: str,
    base_cols: Sequence[str],
    candidate_cols: Sequence[str],
    *,
    judge: ExperimentJudge | None = None,
    baseline_scores: Sequence[float] | None = None,
    n_noise_controls: int = 0,
    noise_uplifts: Sequence[float] | None = None,
    invalid_reasons: Sequence[str] = (),
    review_reasons: Sequence[str] = (),
    minimum_gain: float | None = None,
    random_state: int = 100,
    n_jobs: int = 1,
    extra: dict | None = None,
    test_train_ratio: float | None = None,
) -> ExperimentResult:
    """Score baseline and baseline+candidate on identical folds and judge the paired uplift.

    Validity failures short-circuit: no model is trained for an INVALID experiment.
    """
    judge = judge or ExperimentJudge()
    hib = higher_is_better(metric)
    base_cols = list(dict.fromkeys(base_cols))
    cand_cols = [c for c in candidate_cols if c not in base_cols]
    use_cols = base_cols + cand_cols
    if invalid_reasons:
        verdict = judge.evaluate([], [], higher_is_better=hib, invalid_reasons=invalid_reasons, review_reasons=review_reasons)
        return ExperimentResult(name, metric, hib, base_cols, cand_cols, [], [], verdict, folds.strategy_name, [], dict(extra or {}))
    missing = [c for c in cand_cols if c not in X.columns]
    if missing:
        verdict = judge.evaluate([], [], higher_is_better=hib, invalid_reasons=[f"candidate columns missing: {missing}"])
        return ExperimentResult(name, metric, hib, base_cols, cand_cols, [], [], verdict, folds.strategy_name, [], dict(extra or {}))
    base = list(baseline_scores) if baseline_scores is not None else fold_scores(model_or_factory, X[base_cols], y, folds, metric, n_jobs)
    cand = fold_scores(model_or_factory, X[use_cols], y, folds, metric, n_jobs)
    noise = list(noise_uplifts) if noise_uplifts is not None else []
    if not noise and n_noise_controls > 0:
        noise = noise_control_uplifts(
            model_or_factory, X, y, folds, metric, base_cols, base, n_controls=n_noise_controls, random_state=random_state, n_jobs=n_jobs
        )
    if test_train_ratio is None:
        test_train_ratio = float(np.mean([len(te) / max(1, len(tr)) for tr, te in folds]))
    verdict = judge.evaluate(
        base,
        cand,
        noise_scores=noise or None,
        minimum_gain=minimum_gain,
        higher_is_better=hib,
        invalid_reasons=invalid_reasons,
        review_reasons=review_reasons,
        test_train_ratio=test_train_ratio,
    )
    return ExperimentResult(name, metric, hib, base_cols, cand_cols, base, cand, verdict, folds.strategy_name, noise, dict(extra or {}))


FAMILY_COLUMNS = [
    "feature_family",
    "columns",
    "baseline_scorer",
    "with_family_scorer",
    "scorer_gain",
    "mean_uplift",
    "ci_low",
    "ci_high",
    "positive_share",
    "noise_threshold",
    "required_gain",
    "n_folds",
    "decision",
    "useful",
    "reasons",
]


def feature_family_value(
    model_or_factory,
    X: pd.DataFrame,
    y,
    *,
    cv,
    metric: str,
    feature_families: dict[str, list[str]],
    n_jobs: int = -1,
    judge: ExperimentJudge | None = None,
    n_noise_controls: int = 0,
    random_state: int = 100,
) -> tuple[pd.DataFrame, list[ExperimentResult]] | pd.DataFrame:
    """Marginal information value of each feature family added to a common core.

    ``core`` is the set of columns not assigned to any family. Every family is
    evaluated with the same model factory on identical folds. Returns a frame with
    paired-uplift statistics and the judge's decision (``useful`` is True only for
    KEEP). The list of ExperimentResult objects is available via ``.attrs['experiments']``.
    """
    folds = ensure_foldset(cv, X, y)
    judge = judge or ExperimentJudge()
    assigned = {c for cols in feature_families.values() for c in cols}
    core = [c for c in X.columns if c not in assigned]
    base_cols = core if core else list(X.columns)
    base = fold_scores(model_or_factory, X[base_cols], y, folds, metric, n_jobs)
    noise = (
        noise_control_uplifts(
            model_or_factory, X, y, folds, metric, base_cols, base, n_controls=n_noise_controls, random_state=random_state, n_jobs=n_jobs
        )
        if n_noise_controls > 0
        else None
    )
    rows, experiments = [], []
    for name, cols in feature_families.items():
        valid = [c for c in cols if c in X.columns]
        if not valid:
            continue
        res = run_paired_experiment(
            name,
            model_or_factory,
            X,
            y,
            folds,
            metric,
            base_cols,
            valid,
            judge=judge,
            baseline_scores=base,
            noise_uplifts=noise,
            n_jobs=n_jobs,
        )
        experiments.append(res)
        v = res.verdict
        u = v.uplift
        rows.append(
            {
                "feature_family": name,
                "columns": ",".join(valid),
                "baseline_scorer": v.baseline_mean,
                "with_family_scorer": v.candidate_mean,
                "scorer_gain": u.mean if u else np.nan,
                "mean_uplift": u.mean if u else np.nan,
                "ci_low": u.ci_low if u else np.nan,
                "ci_high": u.ci_high if u else np.nan,
                "positive_share": u.positive_share if u else np.nan,
                "noise_threshold": v.noise_threshold,
                "required_gain": v.required_gain,
                "n_folds": u.n if u else 0,
                "decision": v.decision,
                "useful": v.decision == "KEEP",
                "reasons": "; ".join(v.reasons),
            }
        )
    frame = pd.DataFrame(rows, columns=FAMILY_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("mean_uplift", ascending=False).reset_index(drop=True)
    frame.attrs["experiments"] = experiments
    return frame


def compare_candidate_datasets(
    scientist_factory,
    base_df: pd.DataFrame,
    target: str,
    candidate_feature_sets: dict[str, pd.DataFrame],
    *,
    judge: ExperimentJudge | None = None,
    n_noise_controls: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Controlled experiments for candidate external/new feature sets.

    The scientist is fitted once on ``base_df`` to obtain the champion pipeline and
    the development folds. Each candidate frame (aligned by index) is then judged
    as a paired experiment on those identical folds. The final holdout is never
    touched.
    """
    scientist = scientist_factory()
    fitted = scientist.fit(base_df, target)
    automl = fitted.automl if hasattr(fitted, "automl") else fitted
    cfg = automl.config
    judge = judge or judge_from_config(cfg)
    n_noise = cfg.noise_controls if (n_noise_controls is None and cfg.run_noise_controls) else int(n_noise_controls or 0)
    factory = automl.champion_factory(target)
    folds = automl.folds[target]
    metric = cfg.resolved_metric()
    X_dev, y_dev = automl.development_frame(target, base_df)
    base_cols = list(X_dev.columns)
    base_scores = fold_scores(factory, X_dev, y_dev, folds, metric, n_jobs)
    noise = (
        noise_control_uplifts(
            factory, X_dev, y_dev, folds, metric, base_cols, base_scores, n_controls=n_noise, random_state=cfg.random_state, n_jobs=n_jobs
        )
        if n_noise > 0
        else None
    )
    rows = [
        {
            "experiment": "baseline",
            "decision": "BASELINE",
            "baseline_mean": float(np.mean(base_scores)),
            "candidate_mean": float(np.mean(base_scores)),
            "mean_uplift": 0.0,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "positive_share": np.nan,
            "required_gain": np.nan,
            "noise_threshold": judge.noise_threshold(noise),
            "model": automl.best_row(target).model,
            "validation_strategy": folds.strategy_name,
            "reasons": "",
        }
    ]
    experiments = []
    for name, extra in candidate_feature_sets.items():
        new_cols = [c for c in extra.columns if c not in base_df.columns]
        merged = base_df.join(extra[new_cols], how="left")
        X_cand = merged.loc[X_dev.index, base_cols + new_cols]
        res = run_paired_experiment(
            name,
            factory,
            X_cand,
            y_dev,
            folds,
            metric,
            base_cols,
            new_cols,
            judge=judge,
            baseline_scores=base_scores,
            noise_uplifts=noise,
            n_jobs=n_jobs,
        )
        experiments.append(res)
        v, u = res.verdict, res.verdict.uplift
        rows.append(
            {
                "experiment": name,
                "decision": v.decision,
                "baseline_mean": v.baseline_mean,
                "candidate_mean": v.candidate_mean,
                "mean_uplift": u.mean if u else np.nan,
                "ci_low": u.ci_low if u else np.nan,
                "ci_high": u.ci_high if u else np.nan,
                "positive_share": u.positive_share if u else np.nan,
                "required_gain": v.required_gain,
                "noise_threshold": v.noise_threshold,
                "model": automl.best_row(target).model,
                "validation_strategy": folds.strategy_name,
                "reasons": "; ".join(v.reasons),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["experiments"] = experiments
    return frame


__all__ = [
    "ExperimentResult",
    "FeatureExperiment",
    "asdict",
    "compare_candidate_datasets",
    "ensure_foldset",
    "feature_family_value",
    "fold_scores",
    "judge_from_config",
    "noise_control_uplifts",
    "run_paired_experiment",
]
