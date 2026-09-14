# Changelog

## 0.10.0 — Scientific Integrity & Autonomous Loop
Corrective sprint: the implementation now supports the claims the architecture makes.

### Validation
- `ValidationStrategy` is first-class: `RandomKFold`, `StratifiedKFold`, `GroupKFold`, `RollingOrigin`, `ExpandingWindow`, `SlidingWindow`, configured through `ValidationConfig` (timestamp column, group columns, horizon, gap, min train periods, test periods, step).
- Folds are materialized once (`FoldSet`) and shared by model search, tuning, feature-family experiments, candidate-dataset experiments, noise controls, stability analysis and conformal calibration.
- Temporal strategies never train on observations on/after the validation window; `FoldSet.assert_temporal_integrity` enforces it.
- Grouped temporal problems (store × week) split on the period axis across all groups.

### Locked final holdout
- `FinalHoldout` is reserved before search (last periods for temporal problems), hidden from every experiment, and evaluated exactly once for the champion. Repeat evaluation raises `HoldoutAlreadyEvaluated`; every access attempt is logged (`holdout_access_log.csv`).
- Per-candidate holdout scores are gone; `results.csv` carries development fold scores only. Residual diagnostics and importance use out-of-fold development predictions.

### Experiment Judge
- `ExperimentJudge` evaluates paired fold differences: mean/median/std, positive-fold share, bootstrap confidence interval, worst/best fold.
- Decisions: `KEEP` / `REJECT` / `INCONCLUSIVE` / `INVALID` / `REVIEW`.
- `minimum_feature_gain` (absolute) and `minimum_relative_gain` are enforced; random and permuted noise controls set an empirical noise floor (`noise_controls`, `noise_quantile`).
- `feature_family_value`, `compare_candidate_datasets` and the discovery loop all return judged decisions.

### Feature availability
- `FeatureMetadata` (event time / available time / offset). Availability is `available`, `unavailable` or `unknown`; unknown is never treated as safe (`unknown_availability_policy`).
- Discovery candidates carry real availability instead of a hard-coded temporal validity of 1.0.

### Autonomous loop
- `PredictiveDiscoveryOrchestrator` iterates: diagnose → hypotheses → candidates → validity gates (temporal, join, governance) → paired experiment → judge → memory → adopt. `autonomous_rounds` is the iteration limit.
- `Hypothesis` objects record their evidence; `HypothesisGenerator` uses residual, period, segment, missingness, drift and concept-library signals and can emit `UNKNOWN_SIGNAL`. `HypothesisReasoner` is an optional refinement hook.
- `ExperimentMemory` preserves every result (including INVALID/INCONCLUSIVE) and blocks identical failed experiments without new evidence; saved to `experiment_memory.json`.
- `StopConfig` / recorded stop reasons: max iterations, max experiments, no meaningful improvement, no hypotheses/candidates, target reached, budgets, human stop.

### Value of Information
- `ValueOfInformationEngine.evaluate_experiment` consumes judged experiments; INVALID/INCONCLUSIVE/REJECT produce `NOT_ESTABLISHED_*` with zero value. The portfolio optimizer excludes candidates without established uplift.

### Discovery
- `HybridRelevanceScorer` combines lexical relevance (guardrail) with optional `EmbeddingRelevanceProvider`, graph distance, grain compatibility and experiment-memory priors.
- `JoinValidator` returns `invalid` with reason codes (`ROW_EXPLOSION`, `LOW_COVERAGE`) and the loop refuses to train on such candidates.

### Observability and configuration
- `DiagnosticLog` / `DiagnosticEvent`; silent `except: pass` removed from scientist, autonomous, loop, adapters and analyzers.
- Every `AutoMLConfig` field is consumed; `run_feature_ablation`, `run_noise_controls`, `minimum_feature_gain`, `autonomous_rounds`, `max_ablation_features` and `enable_mlflow` are wired. Unused `explain` and `tuning` extras removed.

### Breaking changes
- `AutoMLResult.results` no longer has `test_performance`; champions have `holdout_performance`. `AutoMLResult.holdout_features` now aliases out-of-fold development rows (`oof_features`).
- `JoinValidation.status` is `pass`/`invalid` (was `pass`/`review`). Discovery-loop rows use `decision` (KEEP/REJECT/INCONCLUSIVE/INVALID/REVIEW) instead of lower-case `status`, and `experiment_runner` should return per-fold `scores`.
- `feature_family_value` returns judged columns (`decision`, `ci_low`, `ci_high`, ...); `useful` is True only for KEEP.

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
