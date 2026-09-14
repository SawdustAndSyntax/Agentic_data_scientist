# Value of Information Agent

Status: **IMPLEMENTED** (`automl_py.value_of_information`, `automl_py.information_portfolio`). Decision-sensitivity / non-linear cost models are **PLANNED**.

## Mission
Translate experimentally established predictive uplift into a transparent economic decision.

## Inputs
- an `ExperimentResult` / `ExperimentVerdict` from the Experiment Judge (not a raw score);
- metric direction;
- user/decision-model supplied value mapping (`BusinessValueModel`);
- license, integration, maintenance, and compute costs (`DataCost`).

## Rules
- Only `KEEP` (and `REVIEW`, flagged) results are valued. `INVALID`, `INCONCLUSIVE` and `REJECT` yield `NOT_ESTABLISHED_*` with zero expected value; `strict=True` raises.
- The paired mean uplift drives expected value; the CI lower bound gives a conservative value.
- The portfolio optimizer only considers candidates with established uplift.

## Outputs
Absolute/relative improvement, expected annual value, first-year and recurring cost and net value, ROI, payback, recommendation (`ACQUIRE_OR_KEEP`, `INVESTIGATE`, `LOW_PRIORITY`, `REJECT`, `PREDICTIVELY_USEFUL_VALUE_UNMODELED`, `NOT_ESTABLISHED_*`, `REVIEW_REQUIRED_*`).

## Must not
- invent dollar value;
- equate predictive uplift with realized business value;
- approve purchases;
- hide assumptions.
