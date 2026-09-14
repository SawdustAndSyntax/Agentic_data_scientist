from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class ExperimentRecommendation:
    priority: int
    experiment: str
    rationale: str
    success_criterion: str
    category: str


class NextExperimentPlanner:
    def plan(
        self,
        *,
        metric: str,
        leakage: pd.DataFrame,
        missingness: pd.DataFrame,
        opportunities: pd.DataFrame,
        drift: dict | None,
        residuals: dict,
        stability: dict | None = None,
        temporal_audit: pd.DataFrame | None = None,
        information_value: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rec = []

        def add(priority, experiment, rationale, criterion, category):
            rec.append(ExperimentRecommendation(priority, experiment, rationale, criterion, category))

        if temporal_audit is not None and not temporal_audit.empty and temporal_audit.temporal_leakage_risk.any():
            bad = ", ".join(temporal_audit.loc[temporal_audit.temporal_leakage_risk, "feature"].head(5))
            add(
                1,
                "Remove future-unavailable features",
                f"Features are not knowable at prediction time: {bad}.",
                "Performance remains acceptable after removing them.",
                "validity",
            )

        if not leakage.empty and (leakage.risk == "high").any():
            bad = ", ".join(leakage.loc[leakage.risk == "high", "feature"].head(5))
            add(
                1,
                "Leakage challenge test",
                f"Potential target/post-outcome leakage detected: {bad}.",
                "CV/holdout performance remains credible after exclusion.",
                "validity",
            )

        if not missingness.empty:
            sys = missingness[missingness.missingness_predictability_auc.fillna(0) >= 0.65]
            if not sys.empty:
                add(
                    2,
                    "Missingness + imputation experiment",
                    "Missingness is systematic for " + ", ".join(sys.feature.head(5)) + ".",
                    "A CV-tested imputer/missing-indicator configuration improves the selected metric.",
                    "data quality",
                )

        high = opportunities[(opportunities.priority == "high") & (~opportunities.present)] if not opportunities.empty else pd.DataFrame()
        if not high.empty:
            fam = ", ".join(high.concept.head(5))
            add(
                3,
                "Acquire/test missing signal families",
                f"Plausible predictive information is absent: {fam}.",
                "Candidate data improves repeated CV and holdout performance beyond the configured minimum gain.",
                "information",
            )

        if drift and drift.get("auc", 0.5) >= 0.65:
            top = ", ".join(drift["feature_importance"].feature.head(5))
            add(
                2,
                "Drift-robust validation",
                f"Train and holdout are distinguishable (AUC={drift['auc']:.3f}); top drivers: {top}.",
                "Performance is stable across time/group-aware or shifted validation.",
                "robustness",
            )

        if stability and stability.get("score_summary", {}).get("coefficient_of_variation", 0) > 0.10:
            cv = stability["score_summary"]["coefficient_of_variation"]
            add(
                2,
                "Stability challenge",
                f"Model score varies materially across resamples (CV={cv:.3f}).",
                "Reduce score variance without materially degrading mean performance.",
                "robustness",
            )

        for target, d in residuals.items():
            if not d.empty:
                f = d.iloc[0].feature
                add(
                    4,
                    f"Residual feature experiment: {f}",
                    f"Remaining error for {target} is associated with {f}.",
                    "Transformations/interactions or upstream variables reduce residual dependence and improve CV.",
                    "feature engineering",
                )

        if information_value is not None and not information_value.empty:
            if "decision" in information_value.columns:
                weak = information_value[information_value.decision.isin(["REJECT"])]
                unclear = information_value[information_value.decision.isin(["INCONCLUSIVE"])]
            else:
                weak = information_value[information_value.scorer_gain <= 0]
                unclear = information_value.iloc[0:0]
            if not weak.empty:
                add(
                    6,
                    "Prune low-value feature families",
                    "Judged REJECT on paired folds: " + ", ".join(weak.feature_family.head(5)) + ".",
                    "Simpler model matches or improves generalization after pruning.",
                    "simplification",
                )
            if not unclear.empty:
                add(
                    5,
                    "Re-test inconclusive feature families",
                    "Uplift CI includes zero for: " + ", ".join(unclear.feature_family.head(5)) + ".",
                    "More folds/periods or a stronger design produces a KEEP or REJECT decision.",
                    "information",
                )

        if not rec:
            add(
                5,
                "Broaden model/tuning search",
                "No major data-validity issue was detected.",
                "Repeated CV improves without increasing instability.",
                "optimization",
            )

        return pd.DataFrame([asdict(x) for x in sorted(rec, key=lambda z: z.priority)])
