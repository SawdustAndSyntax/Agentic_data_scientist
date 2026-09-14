import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from .metrics import scorer_name


def _fresh(model_or_factory, X):
    """A factory ``f(X) -> estimator`` is rebuilt per column subset; a fitted/unfitted estimator is cloned."""
    if callable(model_or_factory) and not hasattr(model_or_factory, "fit"):
        return model_or_factory(X)
    return clone(model_or_factory)


def feature_ablation(model_or_factory, X, y, cv, metric, n_jobs=-1, feature_groups=None):
    """Degradation when each feature group is removed, scored on the shared folds.

    Pass the champion *factory* (``AutoMLResult.champion_factory``) so the preprocessor is
    rebuilt for each column subset; a plain estimator is cloned instead.
    """
    groups = feature_groups or {c: [c] for c in X.columns}
    scorer = scorer_name(metric)
    baseline = float(np.nanmean(cross_val_score(_fresh(model_or_factory, X), X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)))
    rows = []
    for name, cols in groups.items():
        keep = [c for c in X.columns if c not in set(cols)]
        if not keep:
            continue
        score = float(np.nanmean(cross_val_score(_fresh(model_or_factory, X[keep]), X[keep], y, cv=cv, scoring=scorer, n_jobs=n_jobs)))
        rows.append(
            {
                "feature_group": name,
                "baseline_cv_scorer": baseline,
                "without_group_cv_scorer": score,
                "degradation_when_removed": baseline - score,
            }
        )
    return pd.DataFrame(rows).sort_values("degradation_when_removed", ascending=False).reset_index(drop=True)


def add_noise_controls(X, random_state=100, n_random=3):
    out = X.copy()
    rng = np.random.default_rng(random_state)
    for i in range(n_random):
        out[f"__noise_normal_{i + 1}"] = rng.normal(size=len(out))
    if X.shape[1]:
        out["__noise_permuted_feature"] = rng.permutation(X.iloc[:, 0].to_numpy())
    return out
