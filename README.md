# AutoML-Py — Predictive Discovery Engine

> **Find the information your model is missing — and prove whether it is worth using.**

AutoML-Py began as a portable AutoML framework. It is now evolving into **Predictive Discovery**: an autonomous predictive-science system that diagnoses model error, searches governed enterprise semantics for missing signal, validates joins and point-in-time availability, runs controlled experiments, measures marginal predictive value, and translates that uplift into economic Value of Information.

## Why this exists

Conventional AutoML asks:

> *Which model works best on the dataset I gave you?*

Predictive Discovery asks:

> **Is the dataset itself missing information that would materially improve the prediction? Where can we find that information, does it survive a fair experiment, and is it worth the cost?**

## Closed-loop architecture

```text
Prediction objective
  → audit / baseline
  → residual + stability diagnosis
  → missing-signal hypothesis
  → semantic / ontology discovery
  → candidate data + join path
  → temporal + quality validation
  → controlled experiment
  → marginal information value
  → business Value of Information
  → KEEP / ACQUIRE / INVESTIGATE / REJECT
  → next experiment
```

Snowflake and Databricks are adapters into a platform-neutral semantic graph. The differentiated logic stays in the Predictive Discovery layer.

## Documentation

- [`docs/PRODUCT_ARCHITECTURE.md`](docs/PRODUCT_ARCHITECTURE.md) — product thesis, architecture, safety, and end-state behavior.
- [`docs/DIFFERENTIATION.md`](docs/DIFFERENTIATION.md) — positioning vs AutoML, semantic layers, catalogs, feature stores, and analytics agents.
- [`agents/SCIENTIST_AGENT.md`](agents/SCIENTIST_AGENT.md) — predictive scientist contract.
- [`agents/DISCOVERY_AGENT.md`](agents/DISCOVERY_AGENT.md) — governed semantic discovery contract.
- [`agents/EXPERIMENT_AGENT.md`](agents/EXPERIMENT_AGENT.md) — controlled experiment contract.
- [`agents/VALUE_AGENT.md`](agents/VALUE_AGENT.md) — Value of Information contract.
- [`agents/ORCHESTRATOR_AGENT.md`](agents/ORCHESTRATOR_AGENT.md) — closed-loop orchestration contract.

---

## Implementation guide and current capabilities

# AutoML-Py v0.6 — Autonomous Predictive Modeling Scientist

The objective is not "try many algorithms." The objective is:

> Build the strongest defensible out-of-sample prediction possible, determine what information is missing, detect when the evaluation is lying, quantify uncertainty and stability, and recommend the next controlled experiment.

## What v0.6 does

### 1. Data validity
- profile types, cardinality and missingness
- compare imputation strategies inside CV
- add missingness indicators
- detect systematic missingness
- flag likely leakage
- enforce feature availability at prediction time (`FeatureAvailabilityRegistry`)
- adversarial train-vs-holdout drift validation

### 2. Feature discovery
- interactions and PCA
- ratios, group statistics, row statistics
- dates, lags, rolling features
- semantic missing-signal hypotheses (weather, calendar, traffic, events, price/promotion, inventory, macro, competition)
- residual diagnostics
- feature-family information-value experiments
- random/noise controls and ablation helpers

### 3. Model optimization
- linear/regularized models
- RF / ExtraTrees / HistGradientBoosting / SVM / KNN / MLP / trees
- optional XGBoost / LightGBM / CatBoost
- randomized hyperparameter search
- repeated CV champion selection

### 4. Robustness
- repeated-holdout score stability
- feature-importance stability
- drift diagnostics
- split-conformal regression uncertainty intervals

### 5. Autonomous experiment planning
The planner prioritizes validity before optimization:

1. remove unavailable/post-outcome leakage
2. challenge suspicious leakage
3. test missingness/imputation behavior
4. address distribution shift
5. acquire/test missing information families
6. attack residual structure
7. prune low-value feature families
8. only then broaden tuning/model search

It writes `next_experiments.csv` with a rationale and success criterion for every proposed experiment.

## Quick start

```python
from automl_py import (
    AutoMLConfig,
    AutonomousAutoMLScientist,
    FeatureAvailabilityRegistry,
)

config = AutoMLConfig(
    task="regression",
    metric="rmse",
    preprocessors=("original", "scale", "pca"),
    imputation_strategies=("median", "mean", "knn"),
    feature_selection=("none", "mutual_info"),
    interaction_terms=True,
    models=("ridge", "elastic_net", "rf", "extra_trees", "hist_gb"),
    n_jobs=-1,
)

availability = FeatureAvailabilityRegistry(
    {
        "temperature_forecast": "-1d",
        "actual_temperature": "+7d",  # future information -> blocked in strict mode
    }
)

scientist = AutonomousAutoMLScientist(
    config,
    context="weekly store-level ice cream demand",
    feature_availability=availability,
    feature_families={
        "weather": ["temperature_forecast", "humidity"],
        "promotion": ["price", "discount"],
        "calendar": ["holiday", "week_of_year"],
    },
)

result = scientist.fit(df, "sales")
print(result.summary())
print(result.next_experiments())
```

## Proving the value of missing data

Feature discovery only proposes hypotheses. It does **not** claim weather will improve ice-cream demand. When candidate data is available, run a controlled experiment:

```python
from automl_py import compare_candidate_datasets, AutonomousAutoMLScientist

comparison = compare_candidate_datasets(
    lambda: AutonomousAutoMLScientist(config, context="weekly ice cream demand"),
    base_df=df,
    target="sales",
    candidate_feature_sets={
        "weather": weather_features,
        "local_events": event_features,
        "foot_traffic": traffic_features,
    },
)
```

The same pipeline/search framework is rerun, so external data earns its place by improving generalization.

## Feature information value

Pass feature families to `AutonomousAutoMLScientist`. v0.6 evaluates each family using identical CV folds and the selected model pipeline. This answers a stronger question than ordinary feature importance:

> How much does predictive performance improve when this information family is present?

## Uncertainty

Regression runs can fit split-conformal intervals. The output includes a nominal coverage level, interval radius and empirical coverage diagnostic. For high-stakes deployment, reserve a dedicated calibration sample rather than repeatedly reusing a development dataset.

## Temporal leakage / as-of semantics

A good model may be impossible to deploy if it relies on information not known at prediction time.

```python
FeatureAvailabilityRegistry(
    {
        "weather_forecast": "-12h",
        "actual_weather": "+1d",
        "final_invoice": "+14d",
    }
)
```

Strict mode blocks future-unavailable fields before training.

## Artifacts

A v0.6 run can produce:

```text
results.csv
best_results.csv
variable_importance.csv
data_profile.csv
missingness_analysis.csv
leakage_report.csv
feature_opportunities.csv
adversarial_validation.csv
drift_feature_importance.csv
<target>__residual_diagnostics.csv
temporal_availability_audit.csv
stability_scores.csv
feature_stability.csv
stability_summary.json
uncertainty_summary.json
feature_family_value.csv
next_experiments.csv
autonomous_summary.txt
*.joblib
```

## Important boundary

v0.6 can identify that weather, traffic, events or another signal family is worth testing. It does not silently retrieve or purchase external data. Supply candidate feature sets explicitly, then let the controlled experiment decide whether they improve generalization.

## Roadmap beyond v0.6

The architecture is now ready for platform adapters, an Optuna-first global search strategy, MLflow champion/challenger promotion, Snowflake/Snowpark execution, Databricks/Spark execution, dedicated rolling-origin forecasting, and optional agents that can search approved data catalogs for candidate features.

MIT License.

# v0.7 — Predictive Discovery Engine

v0.7 adds a governed data-discovery agent beside the predictive scientist.

The new question is no longer only **"what model performs best on the supplied dataset?"** It is also:

> **"What information already available to the enterprise might materially improve this prediction, how can it be joined safely, and does it actually improve generalization?"**

## Architecture

```text
Predictive Scientist
      │
      │ residual / missing-signal hypothesis
      ▼
Data Discovery Agent
      │
      ├── Snowflake semantic views + catalog
      ├── Databricks Unity Catalog
      └── generic/in-memory adapters
      │
      ▼
Platform-neutral SemanticGraph
      │
      ├── entities
      ├── fields
      ├── metrics
      ├── relationships
      └── join keys
      │
      ▼
Candidate ranking
      │
      ├── semantic relevance
      ├── joinability
      ├── bounded data quality / coverage
      └── temporal validity handoff
      │
      ▼
Join validation
      │
      ├── coverage
      ├── duplicate-key risk
      └── row explosion
      │
      ▼
Controlled predictive experiment
      │
      └── KEEP / REJECT / REVIEW
```

## Snowflake

The adapter reads ordinary `INFORMATION_SCHEMA.TABLES/COLUMNS` and, when available,
Snowflake semantic metadata including semantic tables, dimensions, metrics and relationships.
It converts these objects into the same platform-neutral graph used by the rest of the project.

```python
from automl_py import SnowflakeCatalogAdapter, DataDiscoveryAgent, DiscoveryRequest

adapter = SnowflakeCatalogAdapter(
    connection=conn,
    database="ANALYTICS",
    schemas=["PUBLIC", "EXTERNAL_DATA"],
)

agent = DataDiscoveryAgent(adapter)

candidates = agent.discover(
    DiscoveryRequest(
        target="weekly ice cream sales",
        hypothesis="weather temperature humidity",
        grain=("store_id", "week"),
        context="retail demand forecasting",
    ),
    anchor_entities=["ANALYTICS.PUBLIC.SALES"],
)

print(agent.to_frame(candidates))
```

## Databricks

The Databricks adapter consumes Unity Catalog `INFORMATION_SCHEMA` metadata through the same
interface. No predictive-science logic depends on Databricks-specific objects.

```python
from automl_py import DatabricksCatalogAdapter, DataDiscoveryAgent

adapter = DatabricksCatalogAdapter(
    connection=sql_connection,
    catalog="main",
    schemas=["analytics", "features"],
)
agent = DataDiscoveryAgent(adapter)
```

## Metadata-first and bounded by design

Discovery follows a progressive-access pattern:

1. semantic/catalog metadata
2. relationship and key discovery
3. candidate ranking
4. bounded profiling of only top candidates
5. join validation
6. controlled predictive experiment

The included warehouse adapters issue metadata queries and bounded `SELECT ... LIMIT N` samples.
They do not create, alter, or delete warehouse objects.

## Closed-loop experiment handoff

`PredictiveDiscoveryLoop` deliberately dependency-injects both candidate loading and experiment
execution. This keeps warehouse access, join policy, temporal validation, and predictive testing
separately governable.

```python
loop = PredictiveDiscoveryLoop(agent)

result = loop.run(
    request,
    anchor_entities=["ANALYTICS.PUBLIC.SALES"],
    base_df=training_frame,
    candidate_loader=load_candidate_features,
    experiment_runner=run_controlled_model_experiment,
    top_n=3,
)

print(result.discovery_report)
print(result.tested_candidates)
```

A candidate can be rejected *before* modeling if its join produces low coverage or row explosion.
A candidate that joins safely is still rejected unless the controlled experiment improves the
selected out-of-sample metric.

## Why this is separate from the predictive scientist

The discovery agent answers **where might useful information exist?**
The predictive scientist answers **does that information actually improve a defensible model?**
Keeping those responsibilities separate prevents an LLM/catalog crawler from silently deciding
that semantically plausible data is valuable without experimental evidence.

## Current v0.7 boundary

The repository does **not** yet autonomously purchase Marketplace data, grant itself privileges,
or run unrestricted warehouse queries. External source search, economic value-of-information,
approval workflows, and richer Databricks semantic extraction are natural next layers.


---

# v0.8 — Value of Information

v0.8 adds the economic layer to Predictive Discovery.

The framework can now distinguish:

```text
Predictively useful
        ↓
Economically useful
        ↓
Worth acquiring / maintaining
```

## Example

```python
from automl_py import (
    BusinessValueModel,
    DataCost,
    ValueOfInformationEngine,
)

engine = ValueOfInformationEngine(higher_is_better=False)

weather = engine.evaluate(
    candidate="weather",
    baseline_score=114.0,
    candidate_score=91.0,
    metric="rmse",
    business=BusinessValueModel(
        value_per_error_unit=2500,
        annual_decisions=52,
        realization_rate=0.50,
    ),
    cost=DataCost(
        annual_license_cost=85_000,
        one_time_integration_cost=30_000,
        annual_maintenance_cost=15_000,
    ),
)

print(weather)
```

The framework reports:

- absolute predictive improvement
- percent predictive improvement
- expected annual business value
- first-year and recurring cost
- first-year and recurring net value
- ROI
- estimated payback period
- acquire / investigate / reject recommendation

Business value is **never invented by the library**. A user or downstream
decision model must supply the economic mapping from prediction error to value.

## Information portfolio

After evaluating multiple datasets, choose the best information investments
under a budget:

```python
from automl_py import optimize_information_portfolio

portfolio = optimize_information_portfolio(
    voi_report,
    annual_budget=500_000,
)
```

## External signal scout

The new `ExternalSignalScout` defines a vendor-neutral interface for:

- Snowflake Marketplace
- Databricks Marketplace
- internal data exchanges
- approved third-party data catalogs

The core library does not silently buy, subscribe to, or ingest data.

```python
scout = ExternalSignalScout(
    [
        snowflake_marketplace_provider,
        approved_vendor_catalog_provider,
    ]
)

candidates = scout.search(
    "historical and forecast weather by store location",
    concepts=["temperature", "precipitation", "humidity"],
)
```

Those candidates can then enter the normal Predictive Discovery workflow:

```text
missing-signal hypothesis
        ↓
external catalog search
        ↓
candidate dataset
        ↓
join + temporal validation
        ↓
controlled predictive experiment
        ↓
Value of Information
        ↓
ACQUIRE / KEEP / INVESTIGATE / REJECT
```
