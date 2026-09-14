# Changelog

## 0.9.0 — Product Architecture & Agent Contracts
- Formalized Predictive Discovery product thesis and closed-loop architecture.
- Added differentiation document and explicit non-claims.
- Added Scientist, Discovery, Experiment, Value, and Orchestrator agent contracts.
- Reframed README around Predictive Discovery while preserving implementation documentation.
- Documented autonomy boundaries, stop conditions, and audit requirements.




## 0.8.0 — Value of Information
- Added economic Value-of-Information engine.
- Added explicit business-value and data-cost contracts.
- Added first-year/recurring ROI and payback calculations.
- Added information-portfolio optimization under a budget.
- Added vendor-neutral external catalog / marketplace provider protocol.
- Added External Signal Scout for approved candidate-data discovery.
- External acquisition remains human-controlled; no automatic purchasing.

## 0.7.0 — Predictive Discovery Engine
- Added platform-neutral semantic graph contracts.
- Added governed `DataDiscoveryAgent`.
- Added Snowflake catalog + native Semantic View metadata adapter.
- Added Databricks Unity Catalog adapter.
- Added bounded metadata-first profiling.
- Added semantic/hypothesis candidate ranking.
- Added multi-hop join-path discovery.
- Added join validation for coverage, duplicate-key risk, and row explosion.
- Added `PredictiveDiscoveryLoop` to test discovered data through controlled experiments.
- Added in-memory adapter for local development and deterministic tests.
- Added discovery-agent tests including Snowflake semantic metadata and join rejection.

## 0.6.0 — Autonomous Predictive Modeling Scientist
- Added `AutonomousAutoMLScientist` orchestration.
- Added prioritized next-experiment planning with explicit success criteria.
- Added feature availability / as-of semantics and temporal leakage enforcement.
- Added repeated holdout model and feature stability analysis.
- Added split-conformal regression uncertainty intervals.
- Added feature-family marginal information-value experiments.
- Added controlled comparison of candidate external/new feature datasets.
- Added optional MLflow run helper.
- Preserved v0.3 profiling, missingness, leakage, drift, residual and feature discovery diagnostics.
- Preserved optional boosted model adapters and pipeline-safe preprocessing/search.

## 0.3.0 — AutoML Scientist
- Added data profiling, systematic missingness analysis, multiple imputers, leakage detection,
  semantic feature opportunities, adversarial validation and residual diagnostics.

## 0.2.0 — Feature engineering
- Added interaction terms, group/row statistics, ratios, datetime, lag, rolling and PCA helpers.

## 0.1.0 — Python port
- Initial leakage-safe Python AutoML engine.
