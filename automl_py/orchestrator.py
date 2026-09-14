"""The iterative Predictive Discovery loop.

    while not stop:
        diagnosis  = scientist.diagnose(state)
        hypotheses = generator.generate(diagnosis, memory)
        requests   = source.candidates(hypothesis) for each hypothesis, de-duplicated against memory
        for request in queue:
            gate (availability, join, governance) -> INVALID before training
            result  = paired experiment on identical folds (+ noise controls)
            verdict = judge.evaluate(result)
            memory.record(...)
            KEEP -> adopt columns, rebase baseline scores
        state = update(memory)
    lock final feature set -> evaluate the locked holdout once -> model card

Every termination records its StopReason. Every experiment (including INVALID
and INCONCLUSIVE) is preserved in ExperimentMemory.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .diagnostics import DiagnosticLog
from .discovery import FeatureDiscovery
from .experiments import ExperimentResult, fold_scores, judge_from_config, noise_control_uplifts, run_paired_experiment
from .holdout import HoldoutEvaluation
from .hypotheses import Diagnosis, Hypothesis, HypothesisGenerator, hypotheses_frame
from .judge import KEEP, ExperimentJudge
from .memory import ExperimentMemory
from .metrics import higher_is_better
from .residuals import regression_residual_diagnostics
from .scientist import AutoMLScientist, ScientistResult
from .temporal import UNAVAILABLE, UNKNOWN, FeatureAvailabilityRegistry
from .validation import resolve_columns

# ---- stop conditions --------------------------------------------------------------------------
MAX_ITERATIONS = "MAX_ITERATIONS"
MAX_EXPERIMENTS = "MAX_EXPERIMENTS"
NO_MEANINGFUL_IMPROVEMENT = "NO_MEANINGFUL_IMPROVEMENT"
NO_HYPOTHESES = "NO_DEFENSIBLE_HYPOTHESES"
NO_CANDIDATES = "NO_CANDIDATES_FOR_HYPOTHESES"
TARGET_REACHED = "TARGET_PERFORMANCE_REACHED"
TIME_BUDGET = "COMPUTE_BUDGET_EXHAUSTED"
QUERY_BUDGET = "DATA_QUERY_BUDGET_EXHAUSTED"
HUMAN_STOP = "HUMAN_STOP"
GOVERNANCE_BLOCK = "GOVERNANCE_BLOCK"


@dataclass(slots=True)
class StopConfig:
    max_iterations: int = 10
    max_experiments: int = 50
    no_improvement_rounds: int = 3
    minimum_gain: float | None = None  # overrides the judge's absolute floor when set
    target_performance: float | None = None
    max_seconds: float | None = None
    max_queries: int | None = None


@dataclass
class ExperimentBudget:
    max_experiments: int = 50
    max_queries: int | None = None
    max_seconds: float | None = None
    experiments_run: int = 0
    queries_used: int = 0
    started: float = field(default_factory=perf_counter)

    def charge_experiment(self):
        self.experiments_run += 1

    def charge_query(self, n: int = 1):
        self.queries_used += n

    def exhausted(self) -> str | None:
        if self.experiments_run >= self.max_experiments:
            return MAX_EXPERIMENTS
        if self.max_queries is not None and self.queries_used >= self.max_queries:
            return QUERY_BUDGET
        if self.max_seconds is not None and perf_counter() - self.started >= self.max_seconds:
            return TIME_BUDGET
        return None


# ---- candidates ------------------------------------------------------------------------------
@dataclass
class CandidateFeatureSet:
    """Index-aligned candidate columns plus the provenance needed to gate them."""

    name: str
    frame: pd.DataFrame
    concepts: tuple[str, ...] = ()
    columns: list[str] | None = None
    join_path: list[str] = field(default_factory=list)
    availability: dict[str, str] | None = None  # column -> available | unavailable | unknown
    cost: float = 0.0
    governance_status: str = "approved"
    source: str = "in_memory"
    queries: int = 0

    def feature_columns(self) -> list[str]:
        return list(self.columns) if self.columns else list(self.frame.columns)


@runtime_checkable
class CandidateSource(Protocol):
    def candidates(self, hypothesis: Hypothesis, state: LoopState) -> list[CandidateFeatureSet]: ...


class InMemoryCandidateSource:
    """Candidates keyed by concept; matched against a hypothesis' concept / search concepts."""

    def __init__(self, candidates: Sequence[CandidateFeatureSet]):
        self._candidates = list(candidates)

    def candidates(self, hypothesis: Hypothesis, state: LoopState) -> list[CandidateFeatureSet]:
        if hypothesis.kind == "unknown":
            return []
        wanted = {c.lower() for c in hypothesis.search_concepts} | {hypothesis.concept.lower()}
        wanted |= {tok for w in wanted for tok in re.split(r"[^a-z0-9]+", w) if len(tok) > 3}
        out = []
        for c in self._candidates:
            tags = {t.lower() for t in c.concepts} | {c.name.lower()}
            tags |= {tok for t in tags for tok in re.split(r"[^a-z0-9]+", t) if len(tok) > 3}
            if tags & wanted:
                out.append(c)
        return out


@dataclass
class ExperimentRequest:
    request_id: str
    iteration: int
    hypothesis: Hypothesis
    candidate: CandidateFeatureSet
    fingerprint: str


class ExperimentQueue:
    def __init__(self):
        self._q: deque[ExperimentRequest] = deque()
        self._seen: set[str] = set()

    def push(self, req: ExperimentRequest) -> bool:
        if req.fingerprint in self._seen:
            return False
        self._seen.add(req.fingerprint)
        self._q.append(req)
        return True

    def pop(self) -> ExperimentRequest:
        return self._q.popleft()

    def __len__(self):
        return len(self._q)

    def __bool__(self):
        return bool(self._q)


@dataclass
class LoopState:
    iteration: int
    feature_columns: list[str]
    X: pd.DataFrame
    y: pd.Series
    baseline_scores: list[float]
    noise_uplifts: list[float]
    kept: list[str] = field(default_factory=list)
    rounds_without_improvement: int = 0

    @property
    def baseline_mean(self) -> float:
        return float(np.mean(self.baseline_scores))


@dataclass
class OrchestratorResult:
    target: str
    metric: str
    stop_reason: str
    iterations: int
    memory: ExperimentMemory
    final_feature_columns: list[str]
    kept_candidates: list[str]
    baseline_history: pd.DataFrame
    hypotheses_log: pd.DataFrame
    experiments: list[ExperimentResult]
    holdout_evaluation: HoldoutEvaluation | None
    final_model: object
    scientist: ScientistResult
    model_card: dict
    diagnostics: DiagnosticLog

    @property
    def audit_trail(self) -> pd.DataFrame:
        return self.memory.to_frame()

    def summary(self) -> str:
        lines = [
            "Predictive Discovery loop",
            "=" * 24,
            f"target={self.target} metric={self.metric} iterations={self.iterations} stop={self.stop_reason}",
            f"kept: {', '.join(self.kept_candidates) or 'none'}",
        ]
        if self.holdout_evaluation:
            lines.append(
                f"final holdout {self.metric}={self.holdout_evaluation.score:.4f} on {self.holdout_evaluation.n_rows} rows (evaluated once)"
            )
        lines += ["", self.memory.summary()]
        return "\n".join(lines)


class PredictiveDiscoveryOrchestrator:
    def __init__(
        self,
        config: AutoMLConfig,
        *,
        candidate_source: CandidateSource,
        feature_availability: FeatureAvailabilityRegistry | None = None,
        judge: ExperimentJudge | None = None,
        hypothesis_generator: HypothesisGenerator | None = None,
        stop: StopConfig | None = None,
        memory: ExperimentMemory | None = None,
        diagnostics: DiagnosticLog | None = None,
        prediction_time=None,
        human_stop: Callable[[], bool] | None = None,
        governance_check: Callable[[CandidateFeatureSet], str | None] | None = None,
        min_candidate_coverage: float = 0.70,
        n_jobs: int = 1,
    ):
        self.config = config
        self.source = candidate_source
        self.availability = feature_availability
        self.judge = judge or judge_from_config(config)
        self.generator = hypothesis_generator or HypothesisGenerator()
        self.stop = stop or StopConfig(
            max_iterations=config.autonomous_rounds,
            max_experiments=config.max_experiments,
            no_improvement_rounds=config.no_improvement_rounds,
        )
        self.memory = memory or ExperimentMemory()
        self.diagnostics = diagnostics or DiagnosticLog()
        self.prediction_time = prediction_time
        self.human_stop = human_stop
        self.governance_check = governance_check
        self.min_candidate_coverage = min_candidate_coverage
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame, target: str, context: str = "", scientist_result: ScientistResult | None = None) -> OrchestratorResult:
        c = self.config
        metric = c.resolved_metric()
        hib = higher_is_better(metric)
        df = pd.DataFrame(df)
        if scientist_result is None:
            base_cfg = replace(c, evaluate_holdout=False)
            scientist_result = AutoMLScientist(base_cfg, context).fit(df, target)
        sr = scientist_result
        automl = sr.automl
        self.diagnostics.extend(sr.diagnostics)
        if target not in automl.folds:
            raise RuntimeError("baseline model search produced no champion; see diagnostics")
        holdout = automl.holdouts[target]
        if holdout.evaluated:
            raise RuntimeError(
                "the final holdout was already evaluated before the loop; run the base scientist with evaluate_holdout=False"
            )
        folds = automl.folds[target]
        factory = automl.champion_factory(target)
        X_dev, y_dev = automl.development_frame(target, df)
        vcfg = c.resolved_validation()
        groups, timestamps = resolve_columns(df.loc[X_dev.index], vcfg)
        base_scores = fold_scores(factory, X_dev, y_dev, folds, metric, self.n_jobs)
        noise = self._noise(factory, X_dev, y_dev, folds, metric, list(X_dev.columns), base_scores)
        state = LoopState(0, list(X_dev.columns), X_dev, y_dev, base_scores, noise)
        budget = ExperimentBudget(self.stop.max_experiments, self.stop.max_queries, self.stop.max_seconds)
        history = [{"iteration": 0, "baseline_mean": state.baseline_mean, "n_features": len(state.feature_columns), "event": "baseline"}]
        hyp_log: list[pd.DataFrame] = []
        experiments: list[ExperimentResult] = []
        stop_reason = None

        while True:
            stop_reason = self._check_stop(state, hib, budget)
            if stop_reason:
                break
            diagnosis = self._diagnose(state, factory, folds, metric, sr, timestamps, groups, context, target)
            hyps = self.generator.generate(diagnosis, self.memory)
            hyp_log.append(hypotheses_frame(hyps).assign(iteration=state.iteration + 1))
            if not hyps or all(h.kind == "unknown" for h in hyps):
                stop_reason = NO_HYPOTHESES
                self.diagnostics.record(
                    "orchestrator", "INFO", f"iteration {state.iteration + 1}: no defensible hypotheses; unknown signal"
                )
                break
            queue = ExperimentQueue()
            for h in hyps:
                if h.kind == "unknown":
                    continue
                for cand in self.source.candidates(h, state):
                    cols = [col for col in cand.feature_columns() if col not in state.feature_columns]
                    if not cols:
                        continue
                    fp = ExperimentMemory.fingerprint(cols, folds.strategy_name, state.feature_columns)
                    skip, why = self.memory.should_skip(fp, h.evidence_hash, ExperimentMemory.candidate_fingerprint(cols))
                    if skip:
                        self.diagnostics.record("orchestrator", "INFO", f"skip {cand.name} for {h.hypothesis_id}: {why}")
                        continue
                    queue.push(ExperimentRequest(f"REQ-{state.iteration + 1:02d}-{len(queue) + 1:02d}", state.iteration + 1, h, cand, fp))
            if not queue:
                stop_reason = NO_CANDIDATES
                break
            improved = False
            while queue:
                exhausted = budget.exhausted()
                if exhausted:
                    stop_reason = exhausted
                    break
                req = queue.pop()
                res = self._run_request(req, state, factory, folds, metric)
                experiments.append(res)
                budget.charge_experiment()
                budget.charge_query(req.candidate.queries)
                if res.decision == KEEP:
                    cols = res.candidate_columns
                    state.X = state.X.join(req.candidate.frame[cols], how="left")
                    state.feature_columns = state.feature_columns + cols
                    state.baseline_scores = list(res.candidate_scores)
                    state.noise_uplifts = self._noise(
                        factory, state.X, state.y, folds, metric, state.feature_columns, state.baseline_scores
                    )
                    state.kept.append(req.candidate.name)
                    holdout.with_columns(req.candidate.frame, cols)
                    improved = True
                    history.append(
                        {
                            "iteration": state.iteration + 1,
                            "baseline_mean": state.baseline_mean,
                            "n_features": len(state.feature_columns),
                            "event": f"KEEP {req.candidate.name}",
                        }
                    )
            state.iteration += 1
            if stop_reason:
                break
            state.rounds_without_improvement = 0 if improved else state.rounds_without_improvement + 1

        # ---- lock the final system and evaluate the holdout once -------------------------------
        final_model = factory(state.X[state.feature_columns])
        final_model.fit(state.X[state.feature_columns], state.y)
        final_tag = f"{automl.best_row(target).tag}+{len(state.kept)}kept"
        holdout.lock_champion(final_tag)
        evaluation = holdout.evaluate(final_model, final_tag) if holdout.n_rows else None
        card = {
            "target": target,
            "metric": metric,
            "validation_strategy": folds.strategy_name,
            "n_folds": len(folds),
            "development_rows": len(state.X),
            "holdout_rows": holdout.n_rows,
            "iterations": state.iteration,
            "experiments": len(experiments),
            "stop_reason": stop_reason,
            "kept_candidates": list(state.kept),
            "final_feature_columns": list(state.feature_columns),
            "development_score_mean": state.baseline_mean,
            "holdout_score": None if evaluation is None else evaluation.score,
            "holdout_evaluations": len(holdout.evaluations),
            "champion_configuration": getattr(factory, "configuration", {}),
        }
        result = OrchestratorResult(
            target,
            metric,
            stop_reason or "UNKNOWN",
            state.iteration,
            self.memory,
            list(state.feature_columns),
            list(state.kept),
            pd.DataFrame(history),
            pd.concat(hyp_log, ignore_index=True) if hyp_log else hypotheses_frame([]),
            experiments,
            evaluation,
            final_model,
            sr,
            card,
            self.diagnostics,
        )
        self._save(result, holdout)
        return result

    # ------------------------------------------------------------------------------------------ #
    def _check_stop(self, state: LoopState, hib: bool, budget: ExperimentBudget) -> str | None:
        if self.human_stop and self.human_stop():
            return HUMAN_STOP
        if state.iteration >= self.stop.max_iterations:
            return MAX_ITERATIONS
        if state.rounds_without_improvement >= self.stop.no_improvement_rounds:
            return NO_MEANINGFUL_IMPROVEMENT
        tp = self.stop.target_performance
        if tp is not None and ((hib and state.baseline_mean >= tp) or (not hib and state.baseline_mean <= tp)):
            return TARGET_REACHED
        return budget.exhausted()

    def _noise(self, factory, X, y, folds, metric, base_cols, base_scores) -> list[float]:
        c = self.config
        if not c.run_noise_controls or c.noise_controls <= 0:
            return []
        return noise_control_uplifts(
            factory,
            X,
            y,
            folds,
            metric,
            base_cols,
            base_scores,
            n_controls=c.noise_controls,
            random_state=c.random_state,
            n_jobs=self.n_jobs,
        )

    def _diagnose(self, state, factory, folds, metric, sr: ScientistResult, timestamps, groups, context, target) -> Diagnosis:
        X, y = state.X[state.feature_columns], state.y
        oof = np.full(len(X), np.nan)
        with self.diagnostics.capture("oof_predictions", context=f"iteration {state.iteration + 1}"):
            for tr, te in folds:
                m = factory(X.iloc[tr])
                m.fit(X.iloc[tr], y.iloc[tr])
                oof[te] = m.predict(X.iloc[te])
        scored = np.isfinite(oof)
        rf = X.iloc[np.flatnonzero(scored)].copy()
        rf["actual"] = y.to_numpy()[scored]
        rf["predicted"] = oof[scored]
        rf["residual"] = rf.actual - rf.predicted
        rd = None
        if self.config.task == "regression":
            with self.diagnostics.capture("residual_diagnostics"):
                rd = regression_residual_diagnostics(X.iloc[np.flatnonzero(scored)], rf.actual.to_numpy(), rf.predicted.to_numpy())
        ts = None if timestamps is None else timestamps.iloc[np.flatnonzero(scored)]
        gr = None if groups is None else groups.iloc[np.flatnonzero(scored)]
        return Diagnosis(
            target=target,
            metric=metric,
            baseline_scores=list(state.baseline_scores),
            current_columns=list(state.feature_columns),
            residual_frame=rf,
            residual_diagnostics=rd,
            missingness=sr.missingness,
            leakage=sr.leakage,
            drift=sr.drift,
            timestamps=ts,
            groups=gr,
            context=context,
            feature_opportunities=FeatureDiscovery().opportunities(state.feature_columns, target, context),
        )

    def _gate(self, req: ExperimentRequest, state: LoopState, cols: list[str]) -> tuple[list[str], list[str]]:
        invalid, review = [], []
        cand = req.candidate
        # temporal availability -------------------------------------------------------------
        statuses = {}
        for col in cols:
            if cand.availability and col in cand.availability:
                statuses[col] = cand.availability[col]
            elif self.availability is not None:
                statuses[col] = self.availability.status(col, self.prediction_time)
            else:
                statuses[col] = UNKNOWN
        bad = [k for k, v in statuses.items() if v == UNAVAILABLE]
        unknown = [k for k, v in statuses.items() if v == UNKNOWN]
        if bad:
            invalid.append(f"TEMPORAL_LEAKAGE: not available at prediction time: {', '.join(bad)}")
        if unknown:
            policy = self.config.unknown_availability_policy
            if policy == "strict":
                invalid.append(f"UNKNOWN_AVAILABILITY: {', '.join(unknown)}")
            elif policy == "review":
                review.append(f"availability unknown for: {', '.join(unknown)}")
        # governance ---------------------------------------------------------------------------
        if cand.governance_status not in {"approved", "unknown", ""}:
            invalid.append(f"GOVERNANCE_BLOCK: candidate governance status is '{cand.governance_status}'")
        if self.governance_check:
            why = self.governance_check(cand)
            if why:
                invalid.append(f"GOVERNANCE_BLOCK: {why}")
        # join / coverage on the development index ----------------------------------------------
        frame = cand.frame
        if frame.index.has_duplicates:
            invalid.append(f"ROW_EXPLOSION: candidate '{cand.name}' has duplicate keys ({frame.index.duplicated().sum()} duplicates)")
        else:
            aligned = frame.reindex(state.X.index)[cols]
            coverage = float(aligned.notna().all(axis=1).mean()) if len(aligned) else 0.0
            if coverage < self.min_candidate_coverage:
                invalid.append(f"LOW_COVERAGE: candidate covers {coverage:.1%} of development rows (< {self.min_candidate_coverage:.0%})")
        return invalid, review

    def _run_request(self, req: ExperimentRequest, state: LoopState, factory, folds, metric) -> ExperimentResult:
        cand = req.candidate
        cols = [col for col in cand.feature_columns() if col not in state.feature_columns]
        invalid, review = self._gate(req, state, cols)
        X_cand = state.X.join(cand.frame[cols], how="left") if not invalid else state.X
        res = run_paired_experiment(
            cand.name,
            factory,
            X_cand,
            state.y,
            folds,
            metric,
            state.feature_columns,
            cols,
            judge=self.judge,
            baseline_scores=state.baseline_scores,
            noise_uplifts=state.noise_uplifts or None,
            invalid_reasons=invalid,
            review_reasons=review,
            minimum_gain=self.stop.minimum_gain,
            random_state=self.config.random_state,
            n_jobs=self.n_jobs,
            extra={
                "hypothesis_id": req.hypothesis.hypothesis_id,
                "iteration": req.iteration,
                "request_id": req.request_id,
                "cost": cand.cost,
            },
        )
        self.memory.add(
            iteration=req.iteration,
            hypothesis=req.hypothesis,
            candidate=cand.name,
            candidate_columns=cols,
            decision=res.decision,
            validation_strategy=folds.strategy_name,
            base_columns=state.feature_columns,
            join_path=cand.join_path,
            baseline_scores=res.baseline_scores,
            candidate_scores=res.candidate_scores,
            verdict=res.verdict,
            extra={"request_id": req.request_id, "source": cand.source, "cost": cand.cost},
        )
        self.diagnostics.record(
            "experiment",
            "INFO",
            f"{req.request_id} {cand.name}: {res.verdict.summary()}",
            experiment_id=self.memory.records[-1].experiment_id,
        )
        return res

    def _save(self, result: OrchestratorResult, holdout):
        out = Path(self.config.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.memory.save(out / "experiment_memory.json")
        result.audit_trail.to_csv(out / "experiment_audit_trail.csv", index=False)
        result.hypotheses_log.to_csv(out / "hypotheses.csv", index=False)
        result.baseline_history.to_csv(out / "loop_history.csv", index=False)
        holdout.audit().to_csv(out / "holdout_access_log.csv", index=False)
        (out / "model_card.json").write_text(json.dumps(result.model_card, indent=2, default=str))
        (out / "loop_summary.txt").write_text(result.summary())
        self.diagnostics.to_frame().to_csv(out / "diagnostics.csv", index=False)
