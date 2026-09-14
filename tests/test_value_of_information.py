import pandas as pd

from automl_py.information_portfolio import optimize_information_portfolio
from automl_py.value_of_information import BusinessValueModel, DataCost, ValueOfInformationEngine


def test_voi_lower_is_better():
    engine = ValueOfInformationEngine(higher_is_better=False)
    result = engine.evaluate(
        candidate="weather",
        baseline_score=100.0,
        candidate_score=80.0,
        metric="rmse",
        business=BusinessValueModel(
            value_per_error_unit=1000,
            annual_decisions=100,
            realization_rate=0.5,
        ),
        cost=DataCost(
            annual_license_cost=100_000,
            one_time_integration_cost=50_000,
        ),
    )
    assert result.improvement_absolute == 20.0
    assert result.improvement_pct == 0.2
    assert result.expected_annual_business_value == 1_000_000
    assert result.first_year_net_value == 850_000
    assert result.recommendation == "ACQUIRE_OR_KEEP"


def test_voi_rejects_worse_candidate():
    engine = ValueOfInformationEngine(higher_is_better=False)
    result = engine.evaluate(
        "events",
        100,
        101,
        "rmse",
        BusinessValueModel(value_per_error_unit=1000, annual_decisions=10),
        DataCost(annual_license_cost=100),
    )
    assert result.recommendation == "REJECT"
    assert result.expected_annual_business_value == 0


def test_portfolio_optimizer():
    voi = pd.DataFrame(
        [
            {"candidate": "weather", "first_year_cost": 100, "first_year_net_value": 500},
            {"candidate": "traffic", "first_year_cost": 150, "first_year_net_value": 600},
            {"candidate": "events", "first_year_cost": 50, "first_year_net_value": 20},
        ]
    )
    out = optimize_information_portfolio(voi, annual_budget=200)
    assert out["selected"] == ["traffic", "events"] or out["selected"] == ["weather", "events"]
    # Actual optimum is weather+events=520 vs traffic+events=620, so:
    assert set(out["selected"]) == {"traffic", "events"}
    assert out["total_net_value"] == 620
