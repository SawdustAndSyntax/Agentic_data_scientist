from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from .metrics import scorer_name


@dataclass(frozen=True)
class FeatureExperiment:
    name: str
    columns: tuple[str, ...]
    rationale: str = ""


def feature_family_value(
    model_or_factory, X: pd.DataFrame, y, *, cv, metric: str, feature_families: dict[str, list[str]], n_jobs: int = -1
) -> pd.DataFrame:
    """Measure marginal information value by adding each feature family to a common core.

    `core` is the set of columns not assigned to any family. Every family is evaluated using
    the same model/CV folds, avoiding apples-to-oranges comparisons.
    """
    assigned = {c for cols in feature_families.values() for c in cols}
    core = [c for c in X.columns if c not in assigned]
    scorer = scorer_name(metric)
    base_cols = core if core else list(X.columns)

    def fresh(cols):
        if callable(model_or_factory):
            return model_or_factory(X[cols])
        return clone(model_or_factory)

    base = float(np.nanmean(cross_val_score(fresh(base_cols), X[base_cols], y, cv=cv, scoring=scorer, n_jobs=n_jobs)))
    rows = []
    for name, cols in feature_families.items():
        valid = [c for c in cols if c in X.columns]
        if not valid:
            continue
        use = list(dict.fromkeys(base_cols + valid))
        score = float(np.nanmean(cross_val_score(fresh(use), X[use], y, cv=cv, scoring=scorer, n_jobs=n_jobs)))
        gain = score - base  # sklearn scorers are always higher-is-better, including negative losses
        rows.append(
            {
                "feature_family": name,
                "columns": ",".join(valid),
                "baseline_scorer": base,
                "with_family_scorer": score,
                "scorer_gain": gain,
                "useful": bool(gain > 0),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("scorer_gain", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["feature_family", "columns", "baseline_scorer", "with_family_scorer", "scorer_gain", "useful"])
    )


def compare_candidate_datasets(
    scientist_factory, base_df: pd.DataFrame, target: str, candidate_feature_sets: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Run controlled end-to-end experiments for candidate external/new feature sets.

    Candidate frames must align by index. This is the mechanism for proving that weather,
    events, traffic, etc. actually improve generalization instead of merely sounding plausible.
    """
    rows = []
    baseline = scientist_factory().fit(base_df, target).automl.best_results.iloc[0]
    rows.append(
        {
            "experiment": "baseline",
            "cv_performance": baseline.cv_performance,
            "test_performance": baseline.test_performance,
            "model": baseline.model,
        }
    )
    for name, extra in candidate_feature_sets.items():
        merged = base_df.join(extra, how="left", rsuffix=f"__{name}")
        r = scientist_factory().fit(merged, target).automl.best_results.iloc[0]
        rows.append({"experiment": name, "cv_performance": r.cv_performance, "test_performance": r.test_performance, "model": r.model})
    return pd.DataFrame(rows)
