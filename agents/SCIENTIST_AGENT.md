# Predictive Scientist Agent

Status: **IMPLEMENTED** for baseline modelling, diagnostics and evidence-backed hypothesis generation (`automl_py.scientist`, `automl_py.autonomous`, `automl_py.hypotheses`). LLM-assisted reasoning is **PLANNED** behind the `HypothesisReasoner` hook.

## Mission
Build the strongest defensible out-of-sample prediction while avoiding leakage, unstable signal, and misleading evaluation.

## Inputs
- prediction target, task type and metric;
- dataset plus `ValidationConfig` (timestamp/group columns for temporal or panel problems);
- prediction horizon/context;
- `FeatureAvailabilityRegistry` / `FeatureMetadata`;
- optional feature families, candidate source and business context.

## Responsibilities (as implemented)
1. Audit data quality and missingness.
2. Screen for leakage (name patterns, near-perfect correlation) and enforce feature availability; unknown availability is flagged, never assumed safe.
3. Establish the champion on development folds only; lock the final holdout.
4. Analyze out-of-fold residuals, drift (earliest vs latest development windows), fold stability and conformal uncertainty (chronological calibration for temporal problems).
5. Generate `Hypothesis` objects with recorded evidence from period/segment error patterns, residual–feature association, missingness, drift and absent concept families; emit `UNKNOWN_SIGNAL` when nothing defensible exists.
6. Request experiments through the orchestrator; consult `ExperimentMemory` so failed hypotheses are not re-planned without new evidence.

## Must not
- use final holdout performance to tune or select (enforced by `FinalHoldout`);
- treat correlation as causality;
- silently use post-outcome information;
- fabricate missing-signal explanations;
- purchase or subscribe to data.

## Output
A reproducible modelling report (`autonomous_summary.txt`, `next_experiments.csv`, `hypotheses.csv`, `model_card.json`) plus judged experiment results.
