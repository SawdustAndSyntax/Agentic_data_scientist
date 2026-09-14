from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd


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

    def to_dict(self) -> dict:
        return asdict(self)


class ValueOfInformationEngine:
    """
    Converts experimentally demonstrated predictive uplift into a transparent
    economic case. It does NOT invent business value: the caller must provide
    the value mapping and data costs.
    """

    def __init__(self, higher_is_better: bool):
        self.higher_is_better = higher_is_better

    def evaluate(
        self,
        candidate: str,
        baseline_score: float,
        candidate_score: float,
        metric: str,
        business: BusinessValueModel,
        cost: DataCost | None = None,
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
        )

    def rank(
        self,
        experiments: pd.DataFrame,
        metric: str,
        business_models: Mapping[str, BusinessValueModel],
        costs: Mapping[str, DataCost] | None = None,
        candidate_col: str = "candidate",
        baseline_col: str = "baseline_score",
        candidate_score_col: str = "candidate_score",
    ) -> pd.DataFrame:
        costs = costs or {}
        rows = []

        for _, row in experiments.iterrows():
            name = str(row[candidate_col])
            if name not in business_models:
                continue
            result = self.evaluate(
                candidate=name,
                baseline_score=float(row[baseline_col]),
                candidate_score=float(row[candidate_score_col]),
                metric=metric,
                business=business_models[name],
                cost=costs.get(name),
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
