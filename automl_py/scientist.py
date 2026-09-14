from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .core import AutoML, AutoMLResult
from .diagnostics import DiagnosticLog
from .discovery import FeatureDiscovery
from .drift import adversarial_validation
from .leakage import LeakageDetector
from .missingness import MissingnessAnalyzer
from .profiling import DataProfiler
from .residuals import regression_residual_diagnostics


@dataclass
class ScientistResult:
    automl: AutoMLResult
    data_profile: pd.DataFrame
    missingness: pd.DataFrame
    leakage: pd.DataFrame
    feature_opportunities: pd.DataFrame
    drift: dict | None
    residual_diagnostics: dict
    recommendations: list
    residual_frames: dict = field(default_factory=dict)  # target -> out-of-fold actual/predicted/residual
    diagnostics: DiagnosticLog = field(default_factory=DiagnosticLog)

    def summary(self):
        lines = ["AutoML Scientist summary", "=" * 25]
        for _, r in self.automl.best_results.iterrows():
            hold = f", holdout={r.holdout_performance:.4f}" if np.isfinite(r.holdout_performance) else ""
            lines.append(
                f"- {r.target}: {r.model} / {r.preprocess} / impute={r.imputation} "
                f"({r.validation_strategy} CV={r.cv_performance:.4f} +/- {r.cv_std:.4f}{hold})"
            )
        lines += ["", "Recommended next experiments:"] + [f"- {x}" for x in self.recommendations]
        errs = self.diagnostics.errors()
        if errs:
            lines += ["", f"Diagnostics: {len(errs)} component failure(s) recorded (see diagnostics.csv)"]
        return "\n".join(lines)


class AutoMLScientist:
    """Baseline modelling plus validity diagnostics.

    Residual diagnostics use out-of-fold development predictions; the final
    holdout is never read for planning. Drift is measured between the earliest
    and latest development folds (temporal) or a seeded split of development
    rows (otherwise), again without touching the holdout.
    """

    def __init__(self, config=None, context=""):
        self.config = config or AutoMLConfig()
        self.context = context
        self.result_ = None

    def fit(self, data, target_names):
        c = self.config
        df = pd.DataFrame(data).copy()
        targets = [target_names] if isinstance(target_names, str) else list(target_names)
        t = targets[0]
        diagnostics = DiagnosticLog()
        profile = DataProfiler().profile(df, t) if c.run_data_profile else pd.DataFrame()
        missing = pd.DataFrame()
        if c.run_missingness_analysis:
            with diagnostics.capture("missingness"):
                missing = MissingnessAnalyzer().analyze(df, t, c.random_state)
        leak = LeakageDetector().detect(df, t) if c.run_leakage_detection else pd.DataFrame()
        opp = FeatureDiscovery().opportunities(df.columns, t, self.context)
        auto = AutoML(c).fit(df, targets)
        diagnostics.extend(auto.diagnostics)

        drift = None
        if c.run_adversarial_validation and t in auto.folds:
            with diagnostics.capture("adversarial_validation"):
                X_dev, _ = auto.development_frame(t, df)
                folds = auto.folds[t]
                first_train = folds.folds[0][0]
                last_val = folds.folds[-1][1]
                if folds.is_temporal:
                    a, b = X_dev.iloc[first_train], X_dev.iloc[last_val]
                else:
                    rng = np.random.default_rng(c.random_state)
                    mask = rng.random(len(X_dev)) < 0.7
                    a, b = X_dev.iloc[np.flatnonzero(mask)], X_dev.iloc[np.flatnonzero(~mask)]
                drift = adversarial_validation(a, b, c.random_state)
                drift["comparison"] = "first training window vs last validation window" if folds.is_temporal else "random development split"

        residuals = {}
        residual_frames = {}
        for target in targets:
            br = auto.best_results[auto.best_results.target == target]
            if br.empty:
                continue
            tag = br.iloc[0].tag
            if tag in auto.predictions and tag in auto.oof_features:
                p = auto.predictions[tag]
                frame = auto.oof_features[tag].copy()
                frame["actual"] = p.actual.to_numpy()
                frame["predicted"] = p.predicted.to_numpy()
                frame["residual"] = frame.actual - frame.predicted
                frame["fold"] = p.fold.to_numpy()
                residual_frames[target] = frame
                if c.task == "regression" and c.run_residual_analysis:
                    with diagnostics.capture("residual_diagnostics", context=target):
                        residuals[target] = regression_residual_diagnostics(
                            auto.oof_features[tag], p.actual.to_numpy(), p.predicted.to_numpy()
                        )

        rec = []
        if not leak.empty and (leak.risk == "high").any():
            rec.append("Review potential leakage: " + ", ".join(leak.loc[leak.risk == "high", "feature"].head(5)))
        if not missing.empty and (missing.missingness_predictability_auc.fillna(0) >= 0.65).any():
            rec.append(
                "Treat missingness as signal and compare imputers for: "
                + ", ".join(missing.loc[missing.missingness_predictability_auc.fillna(0) >= 0.65, "feature"].head(5))
            )
        h = opp[(opp.priority == "high") & (~opp.present)]
        if not h.empty:
            rec.append("Test missing/external signal families: " + ", ".join(h.concept.head(5)))
        if drift and drift["auc"] >= 0.65:
            rec.append(
                f"Investigate distribution shift (AUC={drift['auc']:.3f}): " + ", ".join(drift["feature_importance"].feature.head(5))
            )
        for target, d in residuals.items():
            if not d.empty:
                rec.append(
                    f"Residuals for {target} still relate to '{d.iloc[0].feature}'; test nonlinear terms/interactions or an upstream missing driver."
                )
        if not rec:
            rec = ["No major structural issue detected; broaden tuning and validate across additional periods/groups."]
        result = ScientistResult(auto, profile, missing, leak, opp, drift, residuals, rec, residual_frames, diagnostics)
        self.result_ = result
        out = Path(c.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        profile.to_csv(out / "data_profile.csv", index=False)
        missing.to_csv(out / "missingness_analysis.csv", index=False)
        leak.to_csv(out / "leakage_report.csv", index=False)
        opp.to_csv(out / "feature_opportunities.csv", index=False)
        (out / "scientist_recommendations.txt").write_text(result.summary())
        if drift:
            pd.DataFrame([{"auc": drift["auc"], "shift_severity": drift["shift_severity"], "comparison": drift["comparison"]}]).to_csv(
                out / "adversarial_validation.csv", index=False
            )
            drift["feature_importance"].to_csv(out / "drift_feature_importance.csv", index=False)
        for target, d in residuals.items():
            d.to_csv(out / f"{target}__residual_diagnostics.csv", index=False)
        for target, f in residual_frames.items():
            f.to_csv(out / f"{target}__oof_predictions.csv")
        diagnostics.to_frame().to_csv(out / "diagnostics.csv", index=False)
        return result
