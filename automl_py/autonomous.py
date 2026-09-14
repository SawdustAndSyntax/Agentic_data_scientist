from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .ablation import feature_ablation
from .config import AutoMLConfig
from .diagnostics import DiagnosticLog
from .experiments import feature_family_value, judge_from_config
from .orchestrator import OrchestratorResult, PredictiveDiscoveryOrchestrator, StopConfig
from .planner import NextExperimentPlanner
from .scientist import AutoMLScientist, ScientistResult
from .stability import model_stability
from .temporal import FeatureAvailabilityRegistry
from .uncertainty import SplitConformalRegressor


@dataclass
class AutonomousScientistResult:
    scientist: ScientistResult
    experiment_plan: pd.DataFrame
    stability: dict | None
    uncertainty: dict | None
    temporal_audit: pd.DataFrame
    information_value: pd.DataFrame
    ablation: pd.DataFrame | None = None
    loop: OrchestratorResult | None = None
    diagnostics: DiagnosticLog = field(default_factory=DiagnosticLog)

    @property
    def automl(self):
        return self.scientist.automl

    def best_model(self, target):
        return self.scientist.automl.best_model(target)

    def next_experiments(self, n: int = 5):
        return self.experiment_plan.head(n)

    def summary(self):
        parts = [self.scientist.summary(), "", "Autonomous experiment plan", "=" * 26]
        for _, r in self.experiment_plan.head(8).iterrows():
            parts.append(f"P{int(r.priority)} [{r.category}] {r.experiment}: {r.rationale}")
        if not self.information_value.empty:
            parts += ["", "Feature-family information value (paired, judged)"]
            for _, r in self.information_value.iterrows():
                parts.append(
                    f"- {r.feature_family}: {r.decision} uplift={r.mean_uplift:+.4f} CI[{r.ci_low:.4f}, {r.ci_high:.4f}] positive={r.positive_share:.0%}"
                )
        if self.uncertainty:
            parts += [
                "",
                f"Conformal interval: {(1 - self.uncertainty['alpha']) * 100:.0f}% nominal, radius={self.uncertainty['radius']:.4f}, "
                f"empirical coverage={self.uncertainty['empirical_coverage']:.3f} ({self.uncertainty['calibration']})",
            ]
        if self.loop is not None:
            parts += ["", self.loop.summary()]
        errs = self.diagnostics.errors()
        if errs:
            parts += ["", f"{len(errs)} diagnostic error(s) recorded; see diagnostics.csv"]
        return "\n".join(parts)


class AutonomousAutoMLScientist:
    """Single-pass diagnostics plus, when a candidate source is supplied, the real iterative loop.

    It does not fabricate or automatically purchase external data. It identifies
    missing-signal hypotheses, judges candidate feature sets on identical folds
    against noise controls and a practical-significance floor, and records every
    experiment in memory.
    """

    def __init__(
        self,
        config: AutoMLConfig | None = None,
        context: str = "",
        feature_availability: FeatureAvailabilityRegistry | None = None,
        feature_families: dict[str, list[str]] | None = None,
        candidate_source=None,
        stop: StopConfig | None = None,
        prediction_time=None,
    ):
        self.config = config or AutoMLConfig()
        self.context = context
        self.feature_availability = feature_availability or FeatureAvailabilityRegistry()
        self.feature_families = feature_families or {}
        self.candidate_source = candidate_source
        self.stop = stop
        self.prediction_time = prediction_time
        self.result_ = None

    def fit(self, data: pd.DataFrame, target_names: str | list[str]):
        c = self.config
        diagnostics = DiagnosticLog()
        df = pd.DataFrame(data).copy()
        targets = [target_names] if isinstance(target_names, str) else list(target_names)
        target = targets[0]
        vcfg = c.resolved_validation()
        reserved = set(targets) | ({vcfg.timestamp_column} if vcfg.timestamp_column else set()) | set(vcfg.group_columns)
        Xraw = df.drop(columns=[col for col in reserved if col in df.columns])
        temporal_audit = self.feature_availability.audit(Xraw.columns, self.prediction_time)
        unavailable = temporal_audit.loc[temporal_audit.temporal_leakage_risk, "feature"].tolist()
        unknown = temporal_audit.loc[temporal_audit.unknown_availability, "feature"].tolist()
        if unavailable:
            if c.temporal_strict:
                raise ValueError("Temporal leakage risk; unavailable at prediction time: " + ", ".join(unavailable))
            df = df.drop(columns=unavailable)
            diagnostics.record("temporal", "WARNING", f"dropped features unavailable at prediction time: {', '.join(unavailable)}")
        if unknown:
            if c.unknown_availability_policy == "strict":
                raise ValueError("Unknown feature availability (declare in FeatureAvailabilityRegistry): " + ", ".join(unknown))
            diagnostics.record(
                "temporal",
                "WARNING" if c.unknown_availability_policy == "review" else "INFO",
                f"{len(unknown)} feature(s) with UNKNOWN availability kept for review: {', '.join(unknown[:8])}",
            )

        base_cfg = replace(c, evaluate_holdout=False) if self.candidate_source is not None else c
        base = AutoMLScientist(base_cfg, self.context).fit(df, targets)
        diagnostics.extend(base.diagnostics)
        automl = base.automl
        stability = uncertainty = ablation = None
        info = pd.DataFrame()
        judge = judge_from_config(c)
        if target in automl.folds:
            tag = automl.best_row(target).tag
            model = automl.models[tag]
            folds = automl.folds[target]
            X, y = automl.development_frame(target, df)
            factory = automl.champion_factory(target)
            if c.run_stability_analysis:
                with diagnostics.capture("stability"):
                    stability = model_stability(
                        model,
                        X,
                        y,
                        task=c.task,
                        metric=c.resolved_metric(),
                        repeats=c.stability_repeats,
                        test_size=c.test_size,
                        random_state=c.random_state,
                        n_jobs=1,
                        folds=folds,
                        diagnostics=diagnostics,
                    )
            if c.task == "regression" and c.run_uncertainty:
                with diagnostics.capture("uncertainty"):
                    chronological = folds.is_temporal
                    order = np.argsort(folds.timestamps.to_numpy(), kind="stable") if chronological else np.arange(len(X))
                    Xo, yo = X.iloc[order], y.iloc[order]
                    n_eval = max(1, round(len(Xo) * c.test_size))
                    Xfit, Xev, yfit, yev = Xo.iloc[:-n_eval], Xo.iloc[-n_eval:], yo.iloc[:-n_eval], yo.iloc[-n_eval:]
                    conf = SplitConformalRegressor(
                        model, alpha=c.uncertainty_alpha, random_state=c.random_state, chronological=chronological
                    ).fit(Xfit, yfit)
                    uncertainty = {
                        "alpha": c.uncertainty_alpha,
                        "radius": conf.radius_,
                        "empirical_coverage": conf.empirical_coverage(Xev, yev),
                        "calibration": "chronological development split" if chronological else "random development split",
                        "evaluation": "last development rows; the locked holdout is not used",
                    }
            if self.feature_families:
                with diagnostics.capture("feature_family_value"):
                    info = feature_family_value(
                        factory,
                        X,
                        y,
                        cv=folds,
                        metric=c.resolved_metric(),
                        feature_families=self.feature_families,
                        n_jobs=c.n_jobs,
                        judge=judge,
                        n_noise_controls=c.noise_controls if c.run_noise_controls else 0,
                        random_state=c.random_state,
                    )
            if c.run_feature_ablation:
                with diagnostics.capture("feature_ablation"):
                    vi = automl.variable_importance
                    top = (
                        vi[vi.tag == tag].sort_values("importance", ascending=False).variable.head(c.max_ablation_features).tolist()
                        or list(X.columns)[: c.max_ablation_features]
                    )
                    ablation = feature_ablation(
                        factory, X, y, folds, c.resolved_metric(), n_jobs=c.n_jobs, feature_groups={f: [f] for f in top}
                    )

        plan = NextExperimentPlanner().plan(
            metric=c.resolved_metric(),
            leakage=base.leakage,
            missingness=base.missingness,
            opportunities=base.feature_opportunities,
            drift=base.drift,
            residuals=base.residual_diagnostics,
            stability=stability,
            temporal_audit=temporal_audit,
            information_value=info,
        )
        loop = None
        if self.candidate_source is not None and target in automl.folds:
            orchestrator = PredictiveDiscoveryOrchestrator(
                c,
                candidate_source=self.candidate_source,
                feature_availability=self.feature_availability,
                judge=judge,
                stop=self.stop,
                diagnostics=diagnostics,
                prediction_time=self.prediction_time,
                n_jobs=1 if c.n_jobs in (None, 0) else (1 if c.n_jobs < 0 else c.n_jobs),
            )
            loop = orchestrator.run(df, target, self.context, scientist_result=base)
            # surface the final holdout number on the champion row for reporting
            if loop.holdout_evaluation is not None:
                automl.holdout_evaluations[target] = loop.holdout_evaluation
                automl.best_results.loc[automl.best_results.target == target, "holdout_performance"] = loop.holdout_evaluation.score

        result = AutonomousScientistResult(base, plan, stability, uncertainty, temporal_audit, info, ablation, loop, diagnostics)
        self.result_ = result
        out = Path(c.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        plan.to_csv(out / "next_experiments.csv", index=False)
        temporal_audit.to_csv(out / "temporal_availability_audit.csv", index=False)
        if not info.empty:
            info.to_csv(out / "feature_family_value.csv", index=False)
        if ablation is not None:
            ablation.to_csv(out / "feature_ablation.csv", index=False)
        if stability:
            stability["scores"].to_csv(out / "stability_scores.csv", index=False)
            stability["feature_stability"].to_csv(out / "feature_stability.csv", index=False)
            (out / "stability_summary.json").write_text(json.dumps(stability["score_summary"], indent=2))
        if uncertainty:
            (out / "uncertainty_summary.json").write_text(json.dumps(uncertainty, indent=2))
        (out / "autonomous_summary.txt").write_text(result.summary())
        diagnostics.to_frame().to_csv(out / "diagnostics.csv", index=False)
        return result
