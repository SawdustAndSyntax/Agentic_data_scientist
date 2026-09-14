# Predictive Discovery Orchestrator

Status: **IMPLEMENTED** (`automl_py.orchestrator.PredictiveDiscoveryOrchestrator`). Warehouse-backed candidate sources are **EXPERIMENTAL** (the in-memory source is tested; `PredictiveDiscoveryLoop` bridges catalog candidates with injected loaders).

## Loop (as executed)
```python
while not stop:
    diagnosis  = diagnose(current champion, out-of-fold residuals)
    hypotheses = generator.generate(diagnosis, memory)      # evidence-backed, memory-filtered
    queue      = candidates(hypothesis) for each hypothesis  # de-duplicated by fingerprint
    for request in queue:
        gate: availability, join coverage / explosion, governance -> INVALID before training
        result  = paired experiment on identical folds (+ noise controls)
        verdict = judge.evaluate(result)
        memory.record(...)                                   # KEEP, REJECT, INCONCLUSIVE, INVALID, REVIEW
        KEEP -> adopt columns, rebase baseline
lock final feature set -> evaluate the locked holdout once -> model card
```
`AutoMLConfig.autonomous_rounds` is the iteration limit; `StopConfig` refines it.

## Stop conditions (recorded as `stop_reason`)
`MAX_ITERATIONS`, `MAX_EXPERIMENTS`, `NO_MEANINGFUL_IMPROVEMENT`, `NO_DEFENSIBLE_HYPOTHESES`, `NO_CANDIDATES_FOR_HYPOTHESES`, `TARGET_PERFORMANCE_REACHED`, `COMPUTE_BUDGET_EXHAUSTED`, `DATA_QUERY_BUDGET_EXHAUSTED`, `HUMAN_STOP`, `GOVERNANCE_BLOCK`.

## Audit record
`ExperimentMemory` retains, per experiment: hypothesis and evidence, candidate and columns, join path, validation strategy, baseline and candidate fold scores, mean uplift and confidence interval, positive share, noise threshold, required gain, decision and reasons. Persisted to `experiment_memory.json` / `experiment_audit_trail.csv`; the holdout access log proves single evaluation.

## Governing principle
Autonomy applies to scientific reasoning and bounded experimentation. Consequential external actions (purchase, subscribe, spend) remain explicitly governed.
