# Predictive Discovery: Product Architecture

> Implementation status per capability is tracked in the README table. As of 0.10 the closed loop,
> time-aware validation, the locked holdout, the Experiment Judge, experiment memory and
> Value-of-Information gating are implemented and tested; LLM reasoning and warehouse-backed
> candidate sources for the orchestrator remain planned/experimental.

## Product thesis

Predictive Discovery is an autonomous, platform-neutral system for answering a question that conventional AutoML, semantic layers, catalogs, and feature stores do not answer on their own:

> **What information available to the enterprise—or worth acquiring—would materially improve our ability to predict a business outcome, and is that improvement worth the cost?**

The system does not stop at selecting an algorithm. It diagnoses the prediction problem, searches governed semantic metadata for plausible missing signals, validates candidate joins and point-in-time availability, runs controlled experiments, measures incremental predictive value, converts that uplift into Value of Information, and recommends the next experiment.

## The core loop

```text
Prediction objective
      ↓
Data + leakage audit
      ↓
Baseline/champion model
      ↓
Residual / stability / drift diagnosis
      ↓
Missing-signal hypotheses
      ↓
Semantic discovery
      ↓
Candidate data + join paths
      ↓
Temporal / quality / cardinality validation
      ↓
Controlled predictive experiments
      ↓
Marginal information value
      ↓
Business Value of Information
      ↓
KEEP / ACQUIRE / INVESTIGATE / REJECT
      ↓
Next hypothesis
```

## What is differentiated

The moat is **not** metadata crawling, AutoML, an LLM suggesting weather, or a feature store. Those are ingredients.

The differentiated capability is the closed-loop experimental system:

1. **Diagnose** where prediction error remains.
2. **Hypothesize** what information could explain it.
3. **Discover** governed candidate information through enterprise semantics.
4. **Reason over joins** rather than inventing arbitrary SQL relationships.
5. **Enforce as-of validity** so future information cannot leak into training.
6. **Experiment** under identical validation conditions.
7. **Measure marginal information value**, not merely feature importance.
8. **Assess stability and uncertainty** before trusting uplift.
9. **Translate uplift into business value and cost.**
10. **Plan the next experiment** and repeat.

## Architectural layers

```text
┌──────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS SCIENTIST                     │
│ hypothesis generation · experiment planning · recommendations│
└──────────────────────────────┬───────────────────────────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     ▼                         ▼                         ▼
DATA SCIENCE ENGINE     DISCOVERY ENGINE          VALUE ENGINE
profiling               semantic graph            predictive uplift
missingness             catalog adapters          decision sensitivity
leakage                 candidate ranking         business impact
drift                   join planning             data cost
model search            bounded profiling         ROI/payback
residuals               external scout            portfolio selection
stability
uncertainty
     │                         │                         │
     └─────────────────────────┼─────────────────────────┘
                               ▼
                    GOVERNANCE / SAFETY LAYER
                 read-only · RBAC · cost bounds
                 temporal validity · audit trail
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
             Snowflake                   Databricks
       Semantic Views / Catalog     Unity Catalog / Semantics
```

## Platform strategy

Snowflake and Databricks are **adapters**, not the product architecture.

The engine normalizes platform metadata into a shared `SemanticGraph` consisting of entities, fields, relationships, grains, descriptions, temporal metadata, and governance metadata.

This keeps reasoning portable:

```python
catalog = SnowflakeCatalogAdapter(...)
# or
catalog = DatabricksCatalogAdapter(...)

agent = DataDiscoveryAgent(catalog)
```

Everything above the adapter remains platform-neutral.

## Discovery modes

The discovery agent distinguishes four types of signal:

- **Existing signal** — already present in governed enterprise data.
- **Derivable signal** — can be safely engineered from existing information.
- **External signal** — not owned internally, but an approved source may exist.
- **Unknown signal** — residual structure suggests missing information but no defensible candidate is identified yet.

Unknown signal is a valid outcome. The system must not fabricate a causal explanation simply to complete a loop.

## Validation contract

Two controls, both required:

- **Validation strategy** — did the model train only on information that existed before each validation observation? Temporal strategies (`rolling_origin`, `expanding_window`, `sliding_window`) never train on rows on/after the validation window; every experiment shares the same materialized folds.
- **Feature availability** — was this feature knowable when the prediction was made? `available` / `unavailable` / `unknown`; unknown is never treated as safe.

The final holdout is split off before search (the last periods for temporal problems), locked, and evaluated once for the final champion. No candidate, feature, dataset or hyperparameter is ever selected on it.

## Experiment Judge

A candidate is only `KEEP` when, on identical folds, the paired mean uplift clears the larger of the practical floor (`minimum_feature_gain` / `minimum_relative_gain`) and the empirical noise floor (random and permuted control features), the bootstrap confidence interval excludes zero, and the positive-fold share is high enough. Otherwise `REJECT` (even the optimistic bound fails), `INCONCLUSIVE` (uncertain), `INVALID` (temporal, join, leakage or design violation) or `REVIEW` (KEEP-level evidence with an open governance/availability/cost question).

## Candidate evaluation contract

Every candidate should carry:

- semantic relevance;
- join path and relationship confidence;
- expected grain;
- join coverage;
- row-multiplication risk;
- historical coverage;
- missingness;
- point-in-time availability;
- data-quality indicators;
- governance/access status;
- acquisition/maintenance cost where known.

A candidate is not promoted because it sounds plausible. It must survive validation and controlled testing.

## Value of Information

Feature importance asks:

> Which variables did this fitted model use?

Predictive Discovery asks:

> What does having access to this information contribute to out-of-sample prediction?

Value of Information then asks:

> Is that improvement economically worth acquiring, integrating, operating, and governing?

```text
candidate information
      ↓
validated predictive uplift
      ↓
decision sensitivity
      ↓
expected business value
      -
license + integration + maintenance + compute
      ↓
net Value of Information
```

Business value is never invented by the framework. It must come from a user-supplied value model, historical decision analysis, simulation, or another approved decision model.

## Safety principles

The discovery agent is read-only by default.

Permissions should be tiered:

1. **DISCOVER** — metadata and semantic relationships only.
2. **PROFILE** — bounded, approved queries and samples.
3. **EXPERIMENT** — approved extracts/joins within cost and row limits.
4. **ACQUIRE** — always human-controlled unless an organization explicitly implements a separate approval system.

The core framework does not perform DDL, deletes, purchases, or subscriptions.

## What success looks like

A mature run should be able to produce an auditable statement such as:

```text
Objective: predict weekly store demand 7 days ahead.

Baseline RMSE: 114.2

Diagnosis:
- residual error is concentrated in hot-weather periods;
- current dataset contains no weather variables.

Discovery:
- WEATHER_DAILY found through SALES → STORE → LOCATION;
- 98.2% join coverage;
- no material row multiplication;
- forecast temperature is available before prediction time.

Experiment:
- identical rolling validation;
- RMSE 114.2 → 91.6;
- uplift stable across 8 resamples.

Value:
- expected annual decision value: $1.4M;
- annual data + operating cost: $110K;
- recommendation: ACQUIRE / KEEP.

Next experiment:
- test weather × promotion interactions.
```

That is the product.
