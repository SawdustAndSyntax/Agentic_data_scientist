# Experiment Agent

Status: **IMPLEMENTED** (`automl_py.experiments`, `automl_py.judge`, `automl_py.validation`, `automl_py.holdout`).

## Mission
Determine whether candidate information produces stable, out-of-sample predictive improvement under a fair comparison.

## How it works today
- Baseline and candidate are scored on the **identical `FoldSet`** materialized from the configured `ValidationStrategy` (random, stratified, group, rolling-origin, expanding, sliding).
- Paired fold differences feed the **`ExperimentJudge`**: mean, median, std, positive-fold share, Nadeau–Bengio corrected confidence interval and p-value (bootstrap reported alongside), worst/best fold.
- Within one loop iteration all candidates are judged against the same baseline and **Benjamini–Hochberg** control is applied before adoption (forward selection).
- **Noise controls** (random and permuted features on the same folds) set an empirical floor; `minimum_feature_gain` / `minimum_relative_gain` set the practical floor. The larger wins.
- **Validity gates run before training**: temporal availability (`unavailable` → INVALID, `unknown` → review), join validity (`ROW_EXPLOSION`, `LOW_COVERAGE` → INVALID), governance.
- The **final holdout is locked**: never read for candidate comparison, evaluated once for the final champion.

## Decision states
- `KEEP` — mean uplift ≥ required gain, CI lower bound > 0, positive-fold share ≥ threshold, all validity gates passed.
- `REJECT` — valid, but even the upper CI bound cannot reach the required gain.
- `INCONCLUSIVE` — evidence too uncertain (CI includes zero, or too few folds).
- `INVALID` — temporal, join, leakage, coverage or evaluation-design violation.
- `REVIEW` — KEEP-level evidence with an unresolved availability, governance or cost question.

## Must not
- select candidates from repeated final-holdout inspection (enforced: `HoldoutAlreadyEvaluated`);
- compare experiments using inconsistent splits (enforced: the judge returns INVALID for mismatched folds);
- hide failed experiments (every result is recorded in `ExperimentMemory`);
- call feature importance "incremental value".

## Handoff
Returns `ExperimentResult` (scores, verdict, validation strategy, noise uplifts) to the Scientist, memory and the Value Agent.
