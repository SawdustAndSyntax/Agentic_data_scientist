# AutoML-Py — Predictive Discovery Engine

> **Find the information your model is missing — and prove whether it is worth using.**

AutoML-Py began as a portable AutoML framework. It is now a **Predictive Discovery** engine: an autonomous predictive-science system that diagnoses model error, generates evidence-backed hypotheses about missing signal, searches governed semantic metadata for candidates, validates joins and point-in-time availability, runs paired experiments on identical validation folds, judges uplift against a noise floor and a practical-significance floor, and translates *established* uplift into economic Value of Information.

Conventional AutoML asks *which model works best on the dataset I gave you?* Predictive Discovery asks:

> **Is the dataset itself missing information that would materially improve the prediction? Where can we find it, does it survive a fair experiment, and is it worth the cost?**

The core product principle is **the strongest prediction we can defend, not the strongest number we can produce.**

---

## Capability status

Documentation never claims more autonomy or rigour than the code and tests implement.

| Capability | Status | Where |
|---|---|---|
| First-class validation strategies: random, stratified, group, rolling-origin, expanding, sliding windows; grouped temporal (store × week) | **IMPLEMENTED** | `automl_py/validation.py` |
| Identical folds shared by search, tuning, feature-family and candidate experiments, noise controls, stability, conformal calibration | **IMPLEMENTED** | `FoldSet` |
| Locked, single-use final holdout with access log | **IMPLEMENTED** | `automl_py/holdout.py` |
| Paired uplift statistics: Nadeau–Bengio corrected interval and p-value (bootstrap secondary), positive-fold share | **IMPLEMENTED** | `automl_py/judge.py` |
| Benjamini–Hochberg control across candidates per iteration; forward selection | **IMPLEMENTED** | `ExperimentJudge.control_false_discoveries`, `orchestrator.py` |
| Two-stage search screening and a search time budget | **IMPLEMENTED** | `automl_py/core.py` |
| Derived-feature candidates (transforms, interactions, past-only lags/rolling) | **IMPLEMENTED** | `automl_py/sources.py` |
| Point-in-time (as-of) joins with availability lag | **IMPLEMENTED** | `PointInTimeJoiner` |
| Auditable run report and reproducibility manifest | **IMPLEMENTED** | `automl_py/report.py` |
| Noise controls (random and permuted control features) and minimum absolute/relative gain | **IMPLEMENTED** | `ExperimentJudge`, `experiments.py` |
| Decisions `KEEP / REJECT / INCONCLUSIVE / INVALID / REVIEW` | **IMPLEMENTED** | `ExperimentJudge` |
| Feature availability distinct from validation; `available / unavailable / unknown`; unknown never treated as safe | **IMPLEMENTED** | `automl_py/temporal.py` |
| Join validation with `ROW_EXPLOSION` / `LOW_COVERAGE` invalidation before training | **IMPLEMENTED** | `discovery_agent/join_validator.py` |
| Experiment memory with preserved negative results and no-repeat-without-new-evidence | **IMPLEMENTED** | `automl_py/memory.py` |
| Evidence-backed hypothesis generation with `UNKNOWN_SIGNAL` | **IMPLEMENTED** | `automl_py/hypotheses.py` |
| Iterative orchestrator with stop conditions, budgets, audit trail, model card | **IMPLEMENTED** | `automl_py/orchestrator.py` |
| Value of Information consuming judged experiments only; portfolio optimizer | **IMPLEMENTED** | `value_of_information.py`, `information_portfolio.py` |
| Structured diagnostics; no silent exception swallowing | **IMPLEMENTED** | `automl_py/diagnostics.py` |
| Hybrid relevance: lexical guardrail + graph + grain + memory prior; embedding provider hook | **IMPLEMENTED** (embedding provider optional) | `discovery_agent/scoring.py` |
| Snowflake / Databricks catalog adapters | **EXPERIMENTAL** (tested with fake executors, not against live warehouses) | `discovery_agent/adapters/` |
| Discovery-backed candidate source for the orchestrator (catalog → loader → as-of join → gate → judge) | **IMPLEMENTED** against the in-memory catalog; **EXPERIMENTAL** against live warehouses | `DiscoveryCandidateSource` |
| Champion feature ablation on shared folds | **EXPERIMENTAL** (`run_feature_ablation`) | `ablation.py` |
| MLflow logging of the champion | **EXPERIMENTAL** (`enable_mlflow`, `[tracking]` extra) | `tracking.py` |
| LLM-assisted hypothesis reasoning | **EXPERIMENTAL** — `AnthropicHypothesisReasoner` (`[llm]` extra) re-ranks and proposes hypotheses that must cite provided evidence; tested with a fake client, not against the live API | `automl_py/llm.py` |
| Embedding relevance provider | **PLANNED** — hook exists (`EmbeddingRelevanceProvider`); no provider ships | — |
| Decision-sensitivity / non-linear business cost models | **PLANNED** — the value model is linear in metric units | — |
| Automatic external data acquisition | **NOT PLANNED** — search/profile/test/value/recommend only; acquisition stays human-approved | `external_discovery.py` |
| Automatic causal discovery, guaranteed best model | **NOT CLAIMED** | — |

---

## The loop

```text
PREDICTION OBJECTIVE
   ↓
RAW DATA → DEVELOPMENT DATA + FINAL HOLDOUT (locked)
   ↓
validation strategy → identical folds for every experiment
   ↓
baseline champion (development folds only)
   ↓
diagnose: out-of-fold residuals by period / segment / feature, missingness, drift
   ↓
HYPOTHESES (each records its evidence; UNKNOWN_SIGNAL is a valid outcome)
   ↓
CANDIDATES (in-memory, catalog/semantic graph, external scout)
   ↓
VALIDITY GATE: temporal availability · join coverage / explosion · governance   → INVALID before training
   ↓
PAIRED EXPERIMENT on identical folds (+ random / permuted noise controls)
   ↓
EXPERIMENT JUDGE: practical floor · noise floor · confidence interval · positive-fold share
   ↓
KEEP / REJECT / INCONCLUSIVE / INVALID / REVIEW  → EXPERIMENT MEMORY
   ↓
KEEP → adopt columns, rebase baseline → next hypothesis … until a STOP CONDITION
   ↓
LOCK FINAL SYSTEM → FINAL HOLDOUT ONCE → CHAMPION + MODEL CARD
   ↓
VALUE OF INFORMATION (established uplift only) → ACQUIRE / KEEP / INVESTIGATE / LOW PRIORITY
```

---

## Quick start

### Time-aware validation and a judged experiment plan

```python
import pandas as pd
from automl_py import AutoMLConfig, AutonomousAutoMLScientist, FeatureAvailabilityRegistry, ValidationConfig

config = AutoMLConfig(
    task="regression",
    metric="rmse",
    models=("ridge", "hist_gb"),
    validation=ValidationConfig(
        strategy="rolling_origin",  # never trains on observations after the validation window
        timestamp_column="week",
        group_columns=["store_id"],  # store x week: every store moves forward in time together
        min_train_periods=52,
        test_periods=4,
        step=4,
        horizon=1,
        gap=0,
    ),
    test_size=0.2,  # last 20% of periods become the locked final holdout
    minimum_feature_gain=1.0,  # practical-significance floor in metric units
    noise_controls=5,  # random/permuted controls set the empirical noise floor
)

availability = FeatureAvailabilityRegistry(
    {
        "temperature_forecast": "-1d",  # knowable before prediction time
        "actual_temperature": "+7d",  # post-outcome -> blocked in strict mode
    }
)

scientist = AutonomousAutoMLScientist(
    config,
    context="weekly store-level ice cream demand",
    feature_availability=availability,
    feature_families={"weather": ["temperature_forecast", "humidity"], "promotion": ["price", "discount"]},
)
result = scientist.fit(df, "sales")
print(result.summary())
print(result.information_value[["feature_family", "decision", "mean_uplift", "ci_low", "ci_high", "positive_share", "noise_threshold"]])
```

Features not declared in the registry have **UNKNOWN** availability. They are kept and flagged for review by default (`unknown_availability_policy="review"`); set `"strict"` to refuse them.

### The autonomous loop

```python
from automl_py import CandidateFeatureSet, InMemoryCandidateSource, StopConfig

source = InMemoryCandidateSource(
    [
        CandidateFeatureSet(
            "weather_daily", weather_frame, concepts=("weather", "temperature"), availability={"temperature": "available"}, cost=85_000
        ),
        CandidateFeatureSet("foot_traffic", traffic_frame, concepts=("mobility_traffic",), availability={"visits": "unknown"}),
        CandidateFeatureSet("final_invoice", invoice_frame, concepts=("events",), availability={"final_invoice": "unavailable"}),
    ]
)

scientist = AutonomousAutoMLScientist(
    config,
    context="weekly store demand",
    candidate_source=source,
    stop=StopConfig(max_iterations=10, max_experiments=50, no_improvement_rounds=3),
)
result = scientist.fit(df, "sales")
loop = result.loop
print(loop.summary())  # kept candidates, stop reason, single final holdout score
print(loop.audit_trail)  # every experiment: hypothesis, evidence, scores, CI, decision
print(loop.hypotheses_log)  # why each hypothesis existed
print(loop.model_card)
```

The loop actually iterates (`autonomous_rounds` / `StopConfig.max_iterations`), records every result in `ExperimentMemory`, refuses to rerun an identical failed experiment without new evidence, and evaluates the locked holdout exactly once after the feature set is final.

A candidate frame is index-aligned with the training frame. Any candidate source that implements `candidates(hypothesis, state) -> list[CandidateFeatureSet]` can be plugged in (a warehouse-backed source is the natural next layer; see `PredictiveDiscoveryLoop` for the catalog bridge).

### Judging one experiment directly

```python
from automl_py import ExperimentJudge

judge = ExperimentJudge(minimum_absolute_gain=2.0)
verdict = judge.evaluate(
    baseline_scores=[114, 110, 118, 109, 115],
    candidate_scores=[101, 103, 104, 108, 101],
    noise_scores=[0.4, 1.3, 1.8, 0.2, 0.9],  # mean uplift of each noise control
    higher_is_better=False,
)
print(verdict.summary())
# KEEP: mean uplift 9.8000 [4.8000, 13.8000] positive share 100%, required gain 2.0000, noise P95 1.7000
```

A result such as mean +1.2 with CI [-2.1, 4.3] is `INCONCLUSIVE`, not `KEEP`.

### Value of Information from an established result

```python
from automl_py import BusinessValueModel, DataCost, ValueOfInformationEngine, optimize_information_portfolio

engine = ValueOfInformationEngine(higher_is_better=False)
voi = engine.rank_experiments(
    loop.experiments,
    business_models={"weather_daily": BusinessValueModel(value_per_error_unit=1_000, annual_decisions=5_000, realization_rate=0.6)},
    costs={"weather_daily": DataCost(annual_license_cost=85_000, one_time_integration_cost=40_000)},
)
print(optimize_information_portfolio(voi, annual_budget=500_000))
```

`INVALID`, `INCONCLUSIVE` and `REJECT` experiments produce `NOT_ESTABLISHED_*` with zero expected value; they never become acquisition recommendations. Business value is always supplied by the caller — the framework never invents dollars.

---

## Governed discovery (Snowflake / Databricks)

Snowflake and Databricks are **adapters** into a platform-neutral `SemanticGraph`; the differentiated logic lives above them.

```python
from automl_py import DataDiscoveryAgent, DiscoveryRequest, SnowflakeCatalogAdapter, PredictiveDiscoveryLoop, FeatureAvailabilityRegistry

adapter = SnowflakeCatalogAdapter(connection=conn, database="ANALYTICS", schemas=["PUBLIC", "EXTERNAL_DATA"])
agent = DataDiscoveryAgent(adapter)
candidates = agent.discover(
    DiscoveryRequest(target="weekly ice cream sales", hypothesis="weather temperature humidity", grain=("store_id", "week")),
    anchor_entities=["ANALYTICS.PUBLIC.SALES"],
)
print(agent.to_frame(candidates))  # relevance, joinability, availability (available/unavailable/unknown), join path, reasons

loop = PredictiveDiscoveryLoop(agent, feature_availability=FeatureAvailabilityRegistry({...}))
tested = loop.run(request, anchor_entities=[...], base_df=frame, candidate_loader=load, experiment_runner=score_on_shared_folds, top_n=3)
```

Discovery is progressive and cost-aware: semantic metadata → relationships → ranking → bounded profiling of the top candidates → join validation → judged experiment. Each stage can stop a candidate before more expensive access. Adapters only issue metadata queries and bounded `SELECT ... LIMIT N` samples; they never create, alter or delete warehouse objects, and nothing in the framework purchases or subscribes to data.

Catalog entities declare availability through metadata (`availability`, `availability_offset`, or `event_time` + `available_time`). Undeclared availability is `unknown` and routes the experiment to `REVIEW`.

---

## Artifacts

```text
results.csv                       every configuration: development fold scores (no holdout scores)
best_results.csv                  champion per target incl. the single holdout_performance
holdout_access_log.csv            proves the holdout was evaluated once
<target>__oof_predictions.csv     out-of-fold predictions and residuals used for diagnostics
feature_family_value.csv          judged family experiments (decision, CI, noise threshold)
feature_ablation.csv              (when run_feature_ablation)
stability_scores.csv / feature_stability.csv / stability_summary.json
uncertainty_summary.json          conformal radius and empirical coverage
temporal_availability_audit.csv   available / unavailable / unknown per feature
next_experiments.csv              prioritized plan
experiment_memory.json / experiment_audit_trail.csv / hypotheses.csv / loop_history.csv / model_card.json / loop_summary.txt
diagnostics.csv                   every recorded component failure
```

## Documentation

- [`docs/PRODUCT_ARCHITECTURE.md`](docs/PRODUCT_ARCHITECTURE.md) — thesis, architecture, safety, end state.
- [`docs/DIFFERENTIATION.md`](docs/DIFFERENTIATION.md) — positioning and explicit non-claims.
- [`agents/`](agents/) — Scientist, Discovery, Experiment, Value and Orchestrator contracts, each labelled with its implementation status.
- [`CHANGELOG.md`](CHANGELOG.md) — including the 0.10 breaking changes.

## Install

```bash
pip install -e ".[dev]"        # pytest + ruff
pip install -e ".[boost]"      # optional xgboost / lightgbm / catboost
pip install -e ".[tracking]"   # optional MLflow logging
pytest -q && ruff check .
```

Python ≥ 3.11. MIT License.
