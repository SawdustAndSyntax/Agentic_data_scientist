from __future__ import annotations
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from .metrics import evaluate, scorer_name


def model_stability(
    model, X: pd.DataFrame, y, *, task: str, metric: str, repeats: int = 8, test_size: float = 0.2, random_state: int = 100, n_jobs: int = 1
) -> dict:
    """Repeated holdout stability with permutation importance.

    This measures whether score and feature importance remain consistent across different
    train/test partitions. It is a robustness diagnostic, not a replacement for CV.
    """
    score_rows = []
    importance = []
    for i in range(repeats):
        seed = random_state + i
        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
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
        score_rows.append({"repeat": i + 1, "seed": seed, "score": score})
        try:
            pi = permutation_importance(m, Xte, yte, scoring=scorer_name(metric), n_repeats=3, random_state=seed, n_jobs=n_jobs)
            for feature, imp in zip(X.columns, pi.importances_mean):
                importance.append({"repeat": i + 1, "feature": feature, "importance": float(imp)})
        except Exception:
            pass

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
