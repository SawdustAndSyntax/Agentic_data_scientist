import json
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import AutoMLConfig
from .metrics import evaluate, higher_is_better, scorer_name
from .preprocessing import build_feature_selector, build_preprocessor
from .registry import build_estimator, default_models, parameter_space


@dataclass
class AutoMLResult:
    results: pd.DataFrame
    best_results: pd.DataFrame
    models: dict
    variable_importance: pd.DataFrame
    predictions: dict
    holdout_features: dict

    def best_model(self, target):
        r = self.best_results[self.best_results.target == target]
        if r.empty:
            raise KeyError(target)
        return self.models[r.iloc[0].tag]


class AutoML:
    def __init__(self, config=None):
        self.config = config or AutoMLConfig()
        self.result_ = None

    def _cv(self):
        c = self.config
        cls = RepeatedStratifiedKFold if c.task == "classification" else RepeatedKFold
        return cls(n_splits=c.cv_folds, n_repeats=c.cv_repeats, random_state=c.random_state)

    def fit(self, data, target_names):
        c = self.config
        df = pd.DataFrame(data).copy()
        targets = [target_names] if isinstance(target_names, str) else list(target_names)
        Xall = df.drop(columns=targets)
        metric = c.resolved_metric()
        models = list(c.models or default_models(c.task))
        out = Path(c.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        fitted = {}
        preds = {}
        holds = {}
        imps = []
        for target in targets:
            mask = df[target].notna()
            X = Xall.loc[mask].copy()
            y = df.loc[mask, target].copy()
            strat = y if c.task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=c.test_size, random_state=c.random_state, stratify=strat)
            cv = self._cv()
            for prep in c.preprocessors:
                for imp in c.imputation_strategies:
                    for sel in c.feature_selection:
                        for model_name in models:
                            tag = f"{target}__{prep}__imp-{imp}__sel-{sel}__{model_name}"
                            start = perf_counter()
                            err = None
                            warn = None
                            try:
                                pipe = Pipeline(
                                    [
                                        (
                                            "prep",
                                            build_preprocessor(
                                                Xtr,
                                                prep,
                                                c.pca_variance,
                                                c.interaction_terms,
                                                c.interaction_degree,
                                                imp,
                                                c.add_missing_indicators,
                                            ),
                                        ),
                                        ("select", build_feature_selector(c.task, sel, c.feature_selection_k)),
                                        ("model", build_estimator(model_name, c.task, c.random_state)),
                                    ]
                                )
                                space = parameter_space(model_name, c.task)
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    if c.tune and space:
                                        n = min(c.max_candidates_per_model, max(1, int(np.prod([len(v) for v in space.values()]))))
                                        search = RandomizedSearchCV(
                                            pipe,
                                            space,
                                            n_iter=n,
                                            scoring=scorer_name(metric),
                                            cv=cv,
                                            random_state=c.random_state,
                                            n_jobs=c.n_jobs,
                                            refit=True,
                                            error_score=np.nan,
                                        )
                                        search.fit(Xtr, ytr)
                                        fit = search.best_estimator_
                                        raw = float(search.best_score_)
                                        params = search.best_params_
                                    else:
                                        raw = float(
                                            np.nanmean(
                                                cross_val_score(
                                                    pipe, Xtr, ytr, scoring=scorer_name(metric), cv=cv, n_jobs=c.n_jobs, error_score=np.nan
                                                )
                                            )
                                        )
                                        pipe.fit(Xtr, ytr)
                                        fit = pipe
                                        params = {}
                                    if caught:
                                        warn = " | ".join(sorted({str(w.message) for w in caught}))[:3000]
                                cvscore = -raw if metric in {"rmse", "mae"} else raw
                                yp = fit.predict(Xte)
                                prob = (
                                    fit.predict_proba(Xte)
                                    if metric == "roc_auc" and hasattr(fit, "predict_proba")
                                    else (fit.decision_function(Xte) if metric == "roc_auc" and hasattr(fit, "decision_function") else None)
                                )
                                test = evaluate(metric, yte, yp, prob)
                                fitted[tag] = fit
                                preds[tag] = pd.DataFrame({"actual": np.asarray(yte), "predicted": np.asarray(yp)}, index=yte.index)
                                holds[tag] = Xte.copy()
                                if c.save_models:
                                    joblib.dump(fit, out / f"{tag}.joblib")
                                try:
                                    pi = permutation_importance(
                                        fit,
                                        Xte,
                                        yte,
                                        scoring=scorer_name(metric),
                                        n_repeats=3,
                                        random_state=c.random_state,
                                        n_jobs=c.n_jobs,
                                    )
                                    imps.append(
                                        pd.DataFrame(
                                            {
                                                "target": target,
                                                "tag": tag,
                                                "variable": Xte.columns,
                                                "importance": pi.importances_mean,
                                                "importance_std": pi.importances_std,
                                            }
                                        )
                                    )
                                except Exception:
                                    pass
                            except Exception as e:
                                cvscore = np.nan
                                test = np.nan
                                params = {}
                                err = f"{type(e).__name__}: {e}"
                                warn = warn or traceback.format_exc(limit=1).strip()
                            rows.append(
                                {
                                    "target": target,
                                    "preprocess": prep,
                                    "imputation": imp,
                                    "selection": sel,
                                    "model": model_name,
                                    "tag": tag,
                                    "cv_performance": cvscore,
                                    "test_performance": test,
                                    "train_minutes": (perf_counter() - start) / 60,
                                    "best_params": json.dumps(params, default=str),
                                    "error": err,
                                    "warning": warn,
                                }
                            )
        results = pd.DataFrame(rows)
        ok = results[results.error.isna() & results.cv_performance.notna()].copy()
        asc = not higher_is_better(metric)
        best = (
            ok.sort_values(["target", "cv_performance"], ascending=[True, asc])
            .groupby("target", as_index=False)
            .head(1)
            .reset_index(drop=True)
            if not ok.empty
            else ok
        )
        vi = (
            pd.concat(imps, ignore_index=True)
            if imps
            else pd.DataFrame(columns=["target", "tag", "variable", "importance", "importance_std"])
        )
        result = AutoMLResult(results, best, fitted, vi, preds, holds)
        self.result_ = result
        results.to_csv(out / "results.csv", index=False)
        best.to_csv(out / "best_results.csv", index=False)
        vi.to_csv(out / "variable_importance.csv", index=False)
        return result
