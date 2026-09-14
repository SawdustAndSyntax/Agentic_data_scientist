"""Model search on development folds with a locked, single-use final holdout.

Flow per target::

    RAW DATA -> DEVELOPMENT DATA + FINAL HOLDOUT (locked)
    development folds (ValidationStrategy) -> every candidate scored on identical folds
    champion selected by mean development score
    lock -> ONE final holdout evaluation of the champion only

Residual diagnostics and importance come from out-of-fold development
predictions, never from the holdout.
"""

from __future__ import annotations

import json
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from .config import AutoMLConfig
from .diagnostics import DiagnosticLog
from .holdout import FinalHoldout, HoldoutEvaluation
from .metrics import higher_is_better, scorer_name
from .preprocessing import build_feature_selector, build_preprocessor
from .registry import build_estimator, default_models, parameter_space
from .tracking import mlflow_run
from .validation import development_holdout_split, materialize_folds

RESULT_COLUMNS = [
    "target",
    "preprocess",
    "imputation",
    "selection",
    "model",
    "tag",
    "cv_performance",
    "cv_std",
    "cv_scores",
    "n_folds",
    "validation_strategy",
    "stage",
    "screen_score",
    "train_minutes",
    "best_params",
    "error",
    "warning",
]


def _result_row(target, cfg, cv_perf, cv_std, cv_scores, folds, stage, screen_score, minutes, params, err, warn) -> dict:
    prep, imp, sel, model_name = cfg
    return {
        "target": target,
        "preprocess": prep,
        "imputation": imp,
        "selection": sel,
        "model": model_name,
        "tag": f"{target}__{prep}__imp-{imp}__sel-{sel}__{model_name}",
        "cv_performance": cv_perf,
        "cv_std": cv_std,
        "cv_scores": json.dumps(cv_scores),
        "n_folds": len(folds),
        "validation_strategy": folds.strategy_name,
        "stage": stage,
        "screen_score": np.nan if screen_score is None else screen_score,
        "train_minutes": minutes,
        "best_params": json.dumps(params, default=str),
        "error": err,
        "warning": warn,
    }


def to_metric_scale(metric: str, scorer_value: float) -> float:
    """sklearn scorers are higher-is-better; convert neg_* losses back to metric units."""
    return -float(scorer_value) if metric in {"rmse", "mae"} else float(scorer_value)


def pipeline_factory(config: AutoMLConfig, preprocess: str, imputation: str, selection: str, model: str, params: dict | None = None):
    """Return ``factory(X) -> unfitted Pipeline`` reproducing a search configuration.

    Experiments call the factory with the exact feature frame they evaluate so the
    preprocessor is built for those columns.
    """

    def factory(X: pd.DataFrame) -> Pipeline:
        pipe = Pipeline(
            [
                (
                    "prep",
                    build_preprocessor(
                        X,
                        preprocess,
                        config.pca_variance,
                        config.interaction_terms,
                        config.interaction_degree,
                        imputation,
                        config.add_missing_indicators,
                    ),
                ),
                ("select", build_feature_selector(config.task, selection, config.feature_selection_k)),
                ("model", build_estimator(model, config.task, config.random_state)),
            ]
        )
        if params:
            pipe.set_params(**params)
        return pipe

    factory.configuration = {
        "preprocess": preprocess,
        "imputation": imputation,
        "selection": selection,
        "model": model,
        "params": params or {},
    }
    return factory


@dataclass
class AutoMLResult:
    results: pd.DataFrame
    best_results: pd.DataFrame
    models: dict
    variable_importance: pd.DataFrame
    predictions: dict  # champion tag -> out-of-fold development predictions
    oof_features: dict  # champion tag -> development rows aligned with predictions
    folds: dict = field(default_factory=dict)  # target -> FoldSet
    holdouts: dict = field(default_factory=dict)  # target -> FinalHoldout
    holdout_evaluations: dict = field(default_factory=dict)  # target -> HoldoutEvaluation
    development_index: dict = field(default_factory=dict)  # target -> pandas Index of development rows
    feature_columns: dict = field(default_factory=dict)  # target -> list of feature columns
    factories: dict = field(default_factory=dict)  # champion tag -> pipeline factory
    diagnostics: DiagnosticLog = field(default_factory=DiagnosticLog)
    config: AutoMLConfig | None = None

    @property
    def holdout_features(self) -> dict:  # pragma: no cover - backward-compatible alias
        return self.oof_features

    def best_row(self, target) -> pd.Series:
        r = self.best_results[self.best_results.target == target]
        if r.empty:
            raise KeyError(target)
        return r.iloc[0]

    def best_model(self, target):
        return self.models[self.best_row(target).tag]

    def champion_factory(self, target) -> Callable[[pd.DataFrame], Pipeline]:
        return self.factories[self.best_row(target).tag]

    def development_frame(self, target, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Development rows of ``df`` for ``target`` in the order the folds were materialized."""
        idx = self.development_index[target]
        return df.loc[idx, self.feature_columns[target]], df.loc[idx, target]

    def holdout_performance(self, target) -> float | None:
        ev = self.holdout_evaluations.get(target)
        return None if ev is None else ev.score


class AutoML:
    def __init__(self, config=None):
        self.config = config or AutoMLConfig()
        self.result_ = None

    def fit(self, data, target_names):
        c = self.config
        df = pd.DataFrame(data).copy()
        targets = [target_names] if isinstance(target_names, str) else list(target_names)
        vcfg = c.resolved_validation()
        metric = c.resolved_metric()
        models = list(c.models or default_models(c.task))
        out = Path(c.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        diagnostics = DiagnosticLog()
        excluded = set(targets) | ({vcfg.timestamp_column} if vcfg.timestamp_column else set())
        feature_cols = [col for col in df.columns if col not in excluded]
        scorer = get_scorer(scorer_name(metric))

        rows, fitted, preds, oof_X, imps, factories = [], {}, {}, {}, [], {}
        folds_by_target, holdouts, holdout_evals, dev_index = {}, {}, {}, {}
        best_rows = []

        with mlflow_run(c.enable_mlflow, c.mlflow_experiment) as mlflow:
            for target in targets:
                frame = df.loc[df[target].notna()].copy()
                y_all = frame[target]
                dev_pos, hold_pos = development_holdout_split(frame, y_all, vcfg, c.task, c.test_size)
                dev = frame.iloc[dev_pos]
                hold = frame.iloc[hold_pos]
                X_dev, y_dev = dev[feature_cols], dev[target]
                dev_index[target] = dev.index
                folds = materialize_folds(dev, y_dev, vcfg, c.task)
                folds_by_target[target] = folds
                holdout = FinalHoldout(target, metric, hold[feature_cols], hold[target])
                holdouts[target] = holdout
                diagnostics.record(
                    "validation",
                    "INFO",
                    f"{target}: {folds.strategy_name} with {len(folds)} folds on {len(dev)} development rows; {len(hold)} holdout rows locked",
                )

                configs = [
                    (prep, imp, sel, model_name)
                    for prep in c.preprocessors
                    for imp in c.imputation_strategies
                    for sel in c.feature_selection
                    for model_name in models
                ]
                # ---- stage 1: cheap screening on the first fold ------------------------------------
                screen_scores: dict[tuple, float] = {}
                survivors = list(configs)
                if c.screen_configurations and len(configs) > c.screen_top_k:
                    tr0, te0 = folds.folds[0]
                    for cfg in configs:
                        prep, imp, sel, model_name = cfg
                        try:
                            pipe = pipeline_factory(c, prep, imp, sel, model_name)(X_dev.iloc[tr0])
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pipe.fit(X_dev.iloc[tr0], y_dev.iloc[tr0])
                                raw = scorer(pipe, X_dev.iloc[te0], y_dev.iloc[te0])
                            screen_scores[cfg] = to_metric_scale(metric, raw)
                        except Exception as e:
                            screen_scores[cfg] = np.nan
                            diagnostics.error("screening", e, context="__".join(cfg))
                    ranked = sorted(
                        [cfg for cfg in configs if np.isfinite(screen_scores[cfg])],
                        key=lambda cfg: screen_scores[cfg],
                        reverse=higher_is_better(metric),
                    )
                    survivors = ranked[: c.screen_top_k]
                    diagnostics.record(
                        "screening",
                        "INFO",
                        f"{target}: screened {len(configs)} configurations on fold 1; {len(survivors)} advance to full validation",
                    )
                    for cfg in configs:
                        if cfg not in survivors:
                            prep, imp, sel, model_name = cfg
                            rows.append(
                                _result_row(target, cfg, np.nan, np.nan, [], folds, "screened_out", screen_scores[cfg], 0.0, {}, None, None)
                            )
                # ---- stage 2: full validation (+ tuning) on survivors ------------------------------
                search_started = perf_counter()
                for i, cfg in enumerate(survivors):
                    prep, imp, sel, model_name = cfg
                    tag = f"{target}__{prep}__imp-{imp}__sel-{sel}__{model_name}"
                    if c.max_search_seconds is not None and i > 0 and perf_counter() - search_started > c.max_search_seconds:
                        rows.append(
                            _result_row(
                                target, cfg, np.nan, np.nan, [], folds, "skipped_time_budget", screen_scores.get(cfg), 0.0, {}, None, None
                            )
                        )
                        continue
                    start = perf_counter()
                    err = warn = None
                    cv_scores: list[float] = []
                    params: dict = {}
                    try:
                        factory = pipeline_factory(c, prep, imp, sel, model_name)
                        pipe = factory(X_dev)
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
                                    cv=folds,
                                    random_state=c.random_state,
                                    n_jobs=c.n_jobs,
                                    refit=True,
                                    error_score=np.nan,
                                )
                                search.fit(X_dev, y_dev)
                                fit = search.best_estimator_
                                params = dict(search.best_params_)
                                cv_scores = [
                                    to_metric_scale(metric, search.cv_results_[f"split{i}_test_score"][search.best_index_])
                                    for i in range(len(folds))
                                ]
                            else:
                                raw = cross_val_score(
                                    pipe, X_dev, y_dev, scoring=scorer_name(metric), cv=folds, n_jobs=c.n_jobs, error_score=np.nan
                                )
                                cv_scores = [to_metric_scale(metric, s) for s in raw]
                                pipe.fit(X_dev, y_dev)
                                fit = pipe
                            if caught:
                                warn = " | ".join(sorted({str(w.message) for w in caught}))[:3000]
                        fitted[tag] = fit
                        factories[tag] = pipeline_factory(c, prep, imp, sel, model_name, params)
                        if c.save_models:
                            joblib.dump(fit, out / f"{tag}.joblib")
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        warn = warn or traceback.format_exc(limit=1).strip()
                        diagnostics.error("model_search", e, context=tag)
                    finite = [s for s in cv_scores if np.isfinite(s)]
                    rows.append(
                        _result_row(
                            target,
                            cfg,
                            float(np.mean(finite)) if finite else np.nan,
                            float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                            cv_scores,
                            folds,
                            "full",
                            screen_scores.get(cfg),
                            (perf_counter() - start) / 60,
                            params,
                            err,
                            warn,
                        )
                    )
                skipped = [r for r in rows if r["target"] == target and r["stage"] == "skipped_time_budget"]
                if skipped:
                    diagnostics.record(
                        "model_search",
                        "WARNING",
                        f"{target}: {len(skipped)} configuration(s) skipped after {c.max_search_seconds}s search budget",
                    )

                # ---- champion selection on development folds only -------------------------------
                tr = pd.DataFrame([r for r in rows if r["target"] == target])
                ok = tr[(tr.stage == "full") & tr.error.isna() & tr.cv_performance.notna()]
                if ok.empty:
                    diagnostics.record("model_search", "ERROR", f"{target}: no candidate configuration succeeded", error_type="NoChampion")
                    continue
                asc = not higher_is_better(metric)
                champion = ok.sort_values("cv_performance", ascending=asc).iloc[0].to_dict()
                tag = champion["tag"]
                model = fitted[tag]
                holdout.lock_champion(tag)

                # ---- out-of-fold predictions and importance for diagnostics ---------------------
                factory = factories[tag]
                with diagnostics.capture("oof_predictions", context=tag):
                    oof_pred = np.full(len(X_dev), np.nan)
                    oof_fold = np.full(len(X_dev), -1)
                    last_fit = None
                    last_te = None
                    for k, (tr_idx, te_idx) in enumerate(folds):
                        m = factory(X_dev.iloc[tr_idx])
                        m.fit(X_dev.iloc[tr_idx], y_dev.iloc[tr_idx])
                        oof_pred[te_idx] = m.predict(X_dev.iloc[te_idx])
                        oof_fold[te_idx] = k
                        last_fit, last_te = m, te_idx
                    scored = oof_fold >= 0
                    preds[tag] = pd.DataFrame(
                        {"actual": y_dev.to_numpy()[scored], "predicted": oof_pred[scored], "fold": oof_fold[scored]},
                        index=X_dev.index[scored],
                    )
                    oof_X[tag] = X_dev.iloc[np.flatnonzero(scored)].copy()
                    with diagnostics.capture("permutation_importance", context=tag):
                        pi = permutation_importance(
                            last_fit,
                            X_dev.iloc[last_te],
                            y_dev.iloc[last_te],
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
                                    "variable": X_dev.columns,
                                    "importance": pi.importances_mean,
                                    "importance_std": pi.importances_std,
                                    "evaluated_on": "last development fold",
                                }
                            )
                        )

                # ---- single final holdout evaluation --------------------------------------------
                holdout_score = np.nan
                if not holdout.n_rows:
                    diagnostics.record("holdout", "WARNING", f"{target}: no final holdout reserved (test_size=0)")
                elif c.evaluate_holdout:
                    ev: HoldoutEvaluation = holdout.evaluate(model, tag)
                    holdout_evals[target] = ev
                    holdout_score = ev.score
                else:
                    diagnostics.record("holdout", "INFO", f"{target}: holdout evaluation deferred until the feature set is final")
                champion.update(
                    {"holdout_performance": holdout_score, "holdout_rows": holdout.n_rows, "development_rows": len(dev), "metric": metric}
                )
                best_rows.append(champion)
                if mlflow is not None:
                    mlflow.log_params({"champion": tag, "validation": folds.strategy_name, **json.loads(champion["best_params"])})
                    mlflow.log_metric("cv_performance", champion["cv_performance"])
                    if np.isfinite(holdout_score):
                        mlflow.log_metric("holdout_performance", holdout_score)

        results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        best = pd.DataFrame(best_rows, columns=[*RESULT_COLUMNS, "holdout_performance", "holdout_rows", "development_rows", "metric"])
        vi = (
            pd.concat(imps, ignore_index=True)
            if imps
            else pd.DataFrame(columns=["target", "tag", "variable", "importance", "importance_std", "evaluated_on"])
        )
        result = AutoMLResult(
            results,
            best,
            fitted,
            vi,
            preds,
            oof_X,
            folds_by_target,
            holdouts,
            holdout_evals,
            dev_index,
            {t: list(feature_cols) for t in targets},
            factories,
            diagnostics,
            c,
        )
        self.result_ = result
        results.to_csv(out / "results.csv", index=False)
        best.to_csv(out / "best_results.csv", index=False)
        vi.to_csv(out / "variable_importance.csv", index=False)
        pd.concat([h.audit().assign(target=t) for t, h in holdouts.items()], ignore_index=True).to_csv(
            out / "holdout_access_log.csv", index=False
        )
        diagnostics.to_frame().to_csv(out / "diagnostics.csv", index=False)
        return result
