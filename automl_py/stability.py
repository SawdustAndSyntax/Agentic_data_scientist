from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .diagnostics import DiagnosticLog
from .metrics import evaluate, scorer_name
from .validation import FoldSet


def model_stability(
    model,
    X: pd.DataFrame,
    y,
    *,
    task: str,
    metric: str,
    repeats: int = 8,
    test_size: float = 0.2,
    random_state: int = 100,
    n_jobs: int = 1,
    folds: FoldSet | None = None,
    diagnostics: DiagnosticLog | None = None,
) -> dict:
    """Score and permutation-importance stability across resamples.

    When ``folds`` is given (the shared development FoldSet) every resample is one
    of those folds, so temporal problems are never scored on shuffled splits.
    Otherwise repeated seeded holdouts are used, which is only appropriate for
    exchangeable rows.
    """
    diagnostics = diagnostics or DiagnosticLog(warn=False)
    score_rows = []
    importance = []
    if folds is not None:
        partitions = [(i, f"fold{i + 1}", tr, te) for i, (tr, te) in enumerate(folds)]
        source = folds.strategy_name
    else:
        partitions = []
        for i in range(repeats):
            seed = random_state + i
            strat = y if task == "classification" else None
            tr, te, _, _ = train_test_split(range(len(X)), range(len(X)), test_size=test_size, random_state=seed, stratify=strat)
            partitions.append((i, f"seed{seed}", list(tr), list(te)))
        source = "repeated_holdout"
    for i, label, tr, te in partitions:
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        m = clone(model)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        prob = None
        if metric == "roc_auc":
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(Xte)
            elif hasattr(m, "decision_function"):
                prob = m.decision_function(Xte)
        score = evaluate(metric, yte, yp, prob)
        score_rows.append({"repeat": i + 1, "partition": label, "score": score})
        with diagnostics.capture("stability_importance", context=label):
            pi = permutation_importance(m, Xte, yte, scoring=scorer_name(metric), n_repeats=3, random_state=random_state + i, n_jobs=n_jobs)
            for feature, imp in zip(X.columns, pi.importances_mean, strict=True):
                importance.append({"repeat": i + 1, "feature": feature, "importance": float(imp)})

    scores = pd.DataFrame(score_rows)
    imp = pd.DataFrame(importance)
    score_summary = {
        "mean": float(scores.score.mean()),
        "std": float(scores.score.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(scores.score.min()),
        "max": float(scores.score.max()),
        "coefficient_of_variation": float(abs(scores.score.std(ddof=1) / scores.score.mean()))
        if len(scores) > 1 and scores.score.mean() != 0
        else 0.0,
        "source": source,
        "n_partitions": len(scores),
    }
    if imp.empty:
        feature_summary = pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "positive_share", "stability_score"])
    else:
        feature_summary = (
            imp.groupby("feature")
            .importance.agg(importance_mean="mean", importance_std="std", positive_share=lambda s: float((s > 0).mean()))
            .reset_index()
        )
        denom = feature_summary.importance_mean.abs() + feature_summary.importance_std.fillna(0) + 1e-12
        feature_summary["stability_score"] = (feature_summary.importance_mean.abs() / denom) * feature_summary.positive_share
        feature_summary = feature_summary.sort_values(["stability_score", "importance_mean"], ascending=False).reset_index(drop=True)
    return {"scores": scores, "score_summary": score_summary, "feature_stability": feature_summary}
