from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd

from .judge import KEEP, REVIEW

NOT_ESTABLISHED = "NOT_ESTABLISHED"
UNVALIDATED = "UNVALIDATED"
ESTABLISHED_DECISIONS = {KEEP, REVIEW}


class UpliftNotEstablished(ValueError):
    """Raised when economic value is requested for uplift the judge did not establish."""


@dataclass(frozen=True)
class BusinessValueModel:
    """
    Maps predictive improvement into economic value.

    value_per_error_unit:
        Economic value of reducing the selected error metric by one unit.
        This can be estimated from historical decisions, simulation, or a
        downstream decision model.

    annual_decisions:
        Number of prediction-driven decisions per year.

    realization_rate:
        Fraction of theoretical predictive value expected to translate into
        actual business value (0..1).

    annual_compute_cost:
        Incremental annual serving/retraining cost attributable to this feature.
    """

    value_per_error_unit: float = 0.0
    annual_decisions: float = 1.0
    realization_rate: float = 1.0
    annual_compute_cost: float = 0.0

    def validate(self):
        if self.value_per_error_unit < 0:
            raise ValueError("value_per_error_unit must be >= 0")
        if self.annual_decisions < 0:
            raise ValueError("annual_decisions must be >= 0")
        if not 0 <= self.realization_rate <= 1:
            raise ValueError("realization_rate must be between 0 and 1")
        if self.annual_compute_cost < 0:
            raise ValueError("annual_compute_cost must be >= 0")


@dataclass(frozen=True)
class DataCost:
    annual_license_cost: float = 0.0
    one_time_integration_cost: float = 0.0
    annual_maintenance_cost: float = 0.0
    annual_compute_cost: float = 0.0

    @property
    def first_year_cost(self) -> float:
        return self.annual_license_cost + self.one_time_integration_cost + self.annual_maintenance_cost + self.annual_compute_cost

    @property
    def recurring_annual_cost(self) -> float:
        return self.annual_license_cost + self.annual_maintenance_cost + self.annual_compute_cost


@dataclass(frozen=True)
class ValueOfInformationResult:
    candidate: str
    baseline_score: float
    candidate_score: float
    metric: str
    improvement_absolute: float
    improvement_pct: float
    expected_annual_business_value: float
    first_year_cost: float
    recurring_annual_cost: float
    first_year_net_value: float
    recurring_net_value: float
    first_year_roi: float | None
    recurring_roi: float | None
    payback_months: float | None
    recommendation: str
    evidence_decision: str = UNVALIDATED
    conservative_improvement_absolute: float | None = None
    conservative_annual_business_value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ValueOfInformationEngine:
    """
    Converts experimentally demonstrated predictive uplift into a transparent
    economic case. It does NOT invent business value: the caller must provide
    the value mapping and data costs.

    ``evaluate_experiment`` is the governed entry point: it consumes an
    ExperimentResult / ExperimentVerdict and refuses to produce an acquisition
    recommendation unless the judge's decision is KEEP (or REVIEW, which is
    computed but flagged). ``evaluate`` on raw scores is retained for ad-hoc
    what-if analysis and is labelled UNVALIDATED.
    """

    def __init__(self, higher_is_better: bool):
        self.higher_is_better = higher_is_better

    def evaluate_experiment(
        self,
        experiment,
        business: BusinessValueModel,
        cost: DataCost | None = None,
        *,
        candidate: str | None = None,
        metric: str | None = None,
        strict: bool = False,
    ) -> ValueOfInformationResult:
        """Value a judged experiment. ``experiment`` may be an ExperimentResult or an ExperimentVerdict.

        INVALID / INCONCLUSIVE / REJECT results yield a NOT_ESTABLISHED recommendation with
        zero expected value (or raise when ``strict``). Uses the paired mean uplift and also
        reports the conservative value implied by the CI lower bound.
        """
        verdict = getattr(experiment, "verdict", experiment)
        name = candidate or getattr(experiment, "name", None) or "candidate"
        metric = metric or getattr(experiment, "metric", None) or "metric"
        decision = getattr(verdict, "decision", UNVALIDATED)
        uplift = getattr(verdict, "uplift", None)
        business.validate()
        cost = cost or DataCost()
        if decision not in ESTABLISHED_DECISIONS or uplift is None:
            if strict:
                raise UpliftNotEstablished(f"{name}: experiment decision is {decision}; no economic recommendation can be made")
            return ValueOfInformationResult(
                candidate=name,
                baseline_score=float(getattr(verdict, "baseline_mean", float("nan"))),
                candidate_score=float(getattr(verdict, "candidate_mean", float("nan"))),
                metric=metric,
                improvement_absolute=0.0 if uplift is None else float(uplift.mean),
                improvement_pct=0.0,
                expected_annual_business_value=0.0,
                first_year_cost=cost.first_year_cost + business.annual_compute_cost,
                recurring_annual_cost=cost.recurring_annual_cost + business.annual_compute_cost,
                first_year_net_value=-(cost.first_year_cost + business.annual_compute_cost),
                recurring_net_value=-(cost.recurring_annual_cost + business.annual_compute_cost),
                first_year_roi=None,
                recurring_roi=None,
                payback_months=None,
                recommendation=f"{NOT_ESTABLISHED}_{decision}",
                evidence_decision=decision,
            )
        baseline = float(verdict.baseline_mean)
        improvement = float(uplift.mean)
        candidate_score = baseline + improvement if self.higher_is_better else baseline - improvement
        result = self._evaluate_scores(name, baseline, candidate_score, metric, business, cost, evidence_decision=decision)
        conservative = max(0.0, float(uplift.ci_low))
        cons_value = conservative * (business.value_per_error_unit * business.annual_decisions * business.realization_rate)
        result = ValueOfInformationResult(
            **{**result.to_dict(), "conservative_improvement_absolute": conservative, "conservative_annual_business_value": cons_value}
        )
        if decision == REVIEW:
            result = ValueOfInformationResult(**{**result.to_dict(), "recommendation": f"REVIEW_REQUIRED_{result.recommendation}"})
        return result

    def evaluate(
        self,
        candidate: str,
        baseline_score: float,
        candidate_score: float,
        metric: str,
        business: BusinessValueModel,
        cost: DataCost | None = None,
    ) -> ValueOfInformationResult:
        """Ad-hoc valuation of two raw scores. The result is labelled UNVALIDATED because no
        Experiment Judge decision backs it; prefer ``evaluate_experiment``."""
        return self._evaluate_scores(candidate, baseline_score, candidate_score, metric, business, cost, evidence_decision=UNVALIDATED)

    def _evaluate_scores(
        self,
        candidate: str,
        baseline_score: float,
        candidate_score: float,
        metric: str,
        business: BusinessValueModel,
        cost: DataCost | None = None,
        *,
        evidence_decision: str,
    ) -> ValueOfInformationResult:
        business.validate()
        cost = cost or DataCost()

        if self.higher_is_better:
            improvement_abs = candidate_score - baseline_score
            denom = abs(baseline_score)
        else:
            improvement_abs = baseline_score - candidate_score
            denom = abs(baseline_score)

        improvement_pct = improvement_abs / denom if denom > 1e-12 else 0.0

        annual_value = max(0.0, improvement_abs) * (business.value_per_error_unit * business.annual_decisions * business.realization_rate)

        first_cost = cost.first_year_cost + business.annual_compute_cost
        recurring_cost = cost.recurring_annual_cost + business.annual_compute_cost

        first_net = annual_value - first_cost
        recurring_net = annual_value - recurring_cost

        first_roi = first_net / first_cost if first_cost > 0 else None
        recurring_roi = recurring_net / recurring_cost if recurring_cost > 0 else None

        if annual_value > 0 and first_cost > 0:
            payback_months = 12.0 * first_cost / annual_value
        elif first_cost == 0 and annual_value > 0:
            payback_months = 0.0
        else:
            payback_months = None

        if improvement_abs <= 0:
            recommendation = "REJECT"
        elif annual_value <= 0:
            recommendation = "PREDICTIVELY_USEFUL_VALUE_UNMODELED"
        elif first_net > 0 and (payback_months is None or payback_months <= 18):
            recommendation = "ACQUIRE_OR_KEEP"
        elif recurring_net > 0:
            recommendation = "INVESTIGATE"
        else:
            recommendation = "LOW_PRIORITY"

        return ValueOfInformationResult(
            candidate=candidate,
            baseline_score=float(baseline_score),
            candidate_score=float(candidate_score),
            metric=metric,
            improvement_absolute=float(improvement_abs),
            improvement_pct=float(improvement_pct),
            expected_annual_business_value=float(annual_value),
            first_year_cost=float(first_cost),
            recurring_annual_cost=float(recurring_cost),
            first_year_net_value=float(first_net),
            recurring_net_value=float(recurring_net),
            first_year_roi=None if first_roi is None else float(first_roi),
            recurring_roi=None if recurring_roi is None else float(recurring_roi),
            payback_months=None if payback_months is None else float(payback_months),
            recommendation=recommendation,
            evidence_decision=evidence_decision,
        )

    def rank_experiments(
        self,
        experiments,
        business_models: Mapping[str, BusinessValueModel],
        costs: Mapping[str, DataCost] | None = None,
    ) -> pd.DataFrame:
        """Value a list of judged ExperimentResult objects; only KEEP/REVIEW produce economics."""
        costs = costs or {}
        rows = []
        for exp in experiments:
            name = exp.name
            if name not in business_models:
                continue
            rows.append(self.evaluate_experiment(exp, business_models[name], costs.get(name)).to_dict())
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["first_year_net_value", "improvement_pct"], ascending=False).reset_index(drop=True)

    def rank(
        self,
        experiments: pd.DataFrame,
        metric: str,
        business_models: Mapping[str, BusinessValueModel],
        costs: Mapping[str, DataCost] | None = None,
        candidate_col: str = "candidate",
        baseline_col: str = "baseline_score",
        candidate_score_col: str = "candidate_score",
        decision_col: str = "decision",
    ) -> pd.DataFrame:
        """Rank a frame of experiments. When a ``decision`` column is present only KEEP/REVIEW
        rows are valued; rows without a decision are labelled UNVALIDATED."""
        costs = costs or {}
        rows = []

        for _, row in experiments.iterrows():
            name = str(row[candidate_col])
            if name not in business_models:
                continue
            decision = str(row[decision_col]) if decision_col in experiments.columns else UNVALIDATED
            if decision_col in experiments.columns and decision not in ESTABLISHED_DECISIONS:
                continue
            result = self._evaluate_scores(
                candidate=name,
                baseline_score=float(row[baseline_col]),
                candidate_score=float(row[candidate_score_col]),
                metric=metric,
                business=business_models[name],
                cost=costs.get(name),
                evidence_decision=decision,
            )
            rows.append(result.to_dict())

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values(
                ["first_year_net_value", "improvement_pct"],
                ascending=False,
            )
            .reset_index(drop=True)
        )
