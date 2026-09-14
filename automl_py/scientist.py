from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import AutoMLConfig
from .core import AutoML
from .profiling import DataProfiler
from .missingness import MissingnessAnalyzer
from .leakage import LeakageDetector
from .discovery import FeatureDiscovery
from .drift import adversarial_validation
from .residuals import regression_residual_diagnostics


@dataclass
class ScientistResult:
    automl: object
    data_profile: pd.DataFrame
    missingness: pd.DataFrame
    leakage: pd.DataFrame
    feature_opportunities: pd.DataFrame
    drift: dict | None
    residual_diagnostics: dict
    recommendations: list

    def summary(self):
        lines = ["AutoML Scientist summary", "=" * 25]
        for _, r in self.automl.best_results.iterrows():
            lines.append(
                f"- {r.target}: {r.model} / {r.preprocess} / impute={r.imputation} (CV={r.cv_performance:.4f}, test={r.test_performance:.4f})"
            )
        lines += ["", "Recommended next experiments:"] + [f"- {x}" for x in self.recommendations]
        return "\n".join(lines)


class AutoMLScientist:
    def __init__(self, config=None, context=""):
        self.config = config or AutoMLConfig()
        self.context = context
        self.result_ = None

    def fit(self, data, target_names):
        c = self.config
        df = pd.DataFrame(data).copy()
        targets = [target_names] if isinstance(target_names, str) else list(target_names)
        t = targets[0]
        profile = DataProfiler().profile(df, t) if c.run_data_profile else pd.DataFrame()
        missing = MissingnessAnalyzer().analyze(df, t, c.random_state) if c.run_missingness_analysis else pd.DataFrame()
        leak = LeakageDetector().detect(df, t) if c.run_leakage_detection else pd.DataFrame()
        opp = FeatureDiscovery().opportunities(df.columns, t, self.context)
        auto = AutoML(c).fit(df, targets)
        drift = None
        if c.run_adversarial_validation:
            X = df.drop(columns=targets)
            y = df[t]
            m = y.notna()
            X = X.loc[m]
            y = y.loc[m]
            strat = y if c.task == "classification" else None
            Xtr, Xte, _, _ = train_test_split(X, y, test_size=c.test_size, random_state=c.random_state, stratify=strat)
            try:
                drift = adversarial_validation(Xtr, Xte, c.random_state)
            except Exception:
                drift = None
        residuals = {}
        if c.task == "regression" and c.run_residual_analysis:
            for target in targets:
                br = auto.best_results[auto.best_results.target == target]
                if not br.empty:
                    tag = br.iloc[0].tag
                    if tag in auto.predictions and tag in auto.holdout_features:
                        residuals[target] = regression_residual_diagnostics(
                            auto.holdout_features[tag], auto.predictions[tag].actual.to_numpy(), auto.predictions[tag].predicted.to_numpy()
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
        result = ScientistResult(auto, profile, missing, leak, opp, drift, residuals, rec)
        self.result_ = result
        out = Path(c.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        profile.to_csv(out / "data_profile.csv", index=False)
        missing.to_csv(out / "missingness_analysis.csv", index=False)
        leak.to_csv(out / "leakage_report.csv", index=False)
        opp.to_csv(out / "feature_opportunities.csv", index=False)
        (out / "scientist_recommendations.txt").write_text(result.summary())
        if drift:
            pd.DataFrame([{"auc": drift["auc"], "shift_severity": drift["shift_severity"]}]).to_csv(
                out / "adversarial_validation.csv", index=False
            )
            drift["feature_importance"].to_csv(out / "drift_feature_importance.csv", index=False)
        for target, d in residuals.items():
            d.to_csv(out / f"{target}__residual_diagnostics.csv", index=False)
        return result
