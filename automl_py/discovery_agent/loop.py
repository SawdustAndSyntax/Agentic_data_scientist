"""Bridge discovery candidates into judged, validity-gated experiments.

Order of gates for every candidate, before any model is trained:
    1. temporal availability (unavailable -> INVALID TEMPORAL_LEAKAGE; unknown -> review)
    2. join validity (row explosion / low coverage -> INVALID)
    3. governance (blocked -> INVALID)
Then the experiment runner scores baseline and candidate on identical folds and
the Experiment Judge decides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..judge import INVALID, ExperimentJudge
from ..memory import ExperimentMemory
from ..temporal import UNAVAILABLE, UNKNOWN, FeatureAvailabilityRegistry
from .agent import DataDiscoveryAgent
from .contracts import DatasetCandidate, DiscoveryRequest
from .join_validator import JoinValidator

TEMPORAL_LEAKAGE = "TEMPORAL_LEAKAGE"
GOVERNANCE_BLOCK = "GOVERNANCE_BLOCK"


@dataclass
class DiscoveryExperimentResult:
    discovery_report: pd.DataFrame
    tested_candidates: pd.DataFrame


def _scores(outcome: dict) -> list[float]:
    if "scores" in outcome:
        return [float(s) for s in outcome["scores"]]
    if "score" in outcome:
        return [float(outcome["score"])]
    raise KeyError("experiment_runner must return {'scores': [...]} (per fold) or {'score': x}")


class PredictiveDiscoveryLoop:
    """Run judged experiments for the top discovered candidates.

    ``candidate_loader(candidate) -> (extra_frame, left_keys, right_keys)`` and
    ``experiment_runner(df) -> {"scores": [...per fold...], "higher_is_better": bool}``
    are dependency-injected so warehouse extraction and the predictive scientist
    remain independently governable. The runner must score every frame on the same
    folds so the judge can pair them.
    """

    def __init__(
        self,
        discovery_agent: DataDiscoveryAgent,
        *,
        judge: ExperimentJudge | None = None,
        join_validator: JoinValidator | None = None,
        feature_availability: FeatureAvailabilityRegistry | None = None,
        unknown_availability_policy: str = "review",
        governance_check: Callable[[DatasetCandidate], str | None] | None = None,
        memory: ExperimentMemory | None = None,
    ):
        self.agent = discovery_agent
        self.judge = judge or ExperimentJudge()
        self.join_validator = join_validator or JoinValidator()
        self.feature_availability = feature_availability
        self.unknown_policy = unknown_availability_policy
        self.governance_check = governance_check
        self.memory = memory

    def _availability_gate(self, candidate: DatasetCandidate, columns: list[str], prediction_time) -> tuple[list[str], list[str]]:
        invalid, review = [], []
        if candidate.availability == UNAVAILABLE:
            invalid.append(f"{TEMPORAL_LEAKAGE}: candidate '{candidate.entity.qualified_name}' is only available after prediction time")
        elif candidate.availability == UNKNOWN and self.feature_availability is None:
            (invalid if self.unknown_policy == "strict" else review if self.unknown_policy == "review" else []).append(
                f"availability of '{candidate.entity.qualified_name}' is UNKNOWN"
            )
        if self.feature_availability is not None:
            audit = self.feature_availability.audit(columns, prediction_time)
            bad = audit.loc[audit.temporal_leakage_risk, "feature"].tolist()
            unknown = audit.loc[audit.unknown_availability, "feature"].tolist()
            if bad:
                invalid.append(f"{TEMPORAL_LEAKAGE}: features unavailable at prediction time: {', '.join(bad)}")
            if unknown and candidate.availability != "available":
                if self.unknown_policy == "strict":
                    invalid.append(f"availability unknown for: {', '.join(unknown)}")
                elif self.unknown_policy == "review":
                    review.append(f"availability unknown for: {', '.join(unknown)}")
        return invalid, review

    def run(
        self,
        request: DiscoveryRequest,
        *,
        anchor_entities: list[str],
        base_df: pd.DataFrame,
        candidate_loader,
        experiment_runner,
        top_n: int = 3,
        prediction_time=None,
        iteration: int = 0,
        hypothesis=None,
    ) -> DiscoveryExperimentResult:
        candidates = self.agent.discover(request, anchor_entities=anchor_entities, profile=True)
        report = self.agent.to_frame(candidates)
        rows = []
        baseline = experiment_runner(base_df)
        base_scores = _scores(baseline)
        higher = bool(baseline.get("higher_is_better", True))

        def row(c, decision, reason_code, verdict=None, jv=None, reasons=()):
            u = getattr(verdict, "uplift", None)
            return {
                "candidate": c.entity.qualified_name,
                "decision": decision,
                "reason_code": reason_code,
                "availability": c.availability,
                "baseline_mean": float(pd.Series(base_scores).mean()),
                "candidate_mean": None if verdict is None else verdict.candidate_mean,
                "mean_uplift": None if u is None else u.mean,
                "ci_low": None if u is None else u.ci_low,
                "ci_high": None if u is None else u.ci_high,
                "positive_share": None if u is None else u.positive_share,
                "required_gain": None if verdict is None else verdict.required_gain,
                "join_coverage": None if jv is None else jv.matched_left_pct,
                "row_multiplier": None if jv is None else jv.row_multiplier,
                "join_path": " -> ".join(c.join_plan.path) if c.join_plan else "",
                "reasons": "; ".join(reasons) if reasons else ("; ".join(verdict.reasons) if verdict is not None else ""),
            }

        for c in candidates[:top_n]:
            try:
                extra, left_keys, right_keys = candidate_loader(c)
                left_keys, right_keys = list(left_keys), list(right_keys)
                new_cols = [col for col in extra.columns if col not in right_keys]
                invalid, review = self._availability_gate(c, new_cols, prediction_time)
                gov = self.governance_check(c) if self.governance_check else None
                if gov:
                    invalid.append(f"{GOVERNANCE_BLOCK}: {gov}")
                jv = self.join_validator.validate_frames(base_df, extra, left_keys, right_keys)
                if not jv.valid:
                    invalid.extend(
                        f"{code}: {w}"
                        for code, w in zip(jv.reason_codes, jv.warnings, strict=False)
                        if code in {"ROW_EXPLOSION", "LOW_COVERAGE"}
                    )
                if invalid:
                    verdict = self.judge.evaluate([], [], invalid_reasons=invalid, review_reasons=review)
                    code = invalid[0].split(":")[0]
                    rows.append(row(c, INVALID, code, verdict, jv, invalid))
                    self._remember(iteration, hypothesis, c, new_cols, verdict, base_scores, [])
                    continue
                merged = base_df.merge(extra, left_on=left_keys, right_on=right_keys, how="left", suffixes=("", "__candidate"))
                outcome = experiment_runner(merged)
                cand_scores = _scores(outcome)
                verdict = self.judge.evaluate(
                    base_scores,
                    cand_scores,
                    noise_scores=outcome.get("noise_uplifts"),
                    higher_is_better=bool(outcome.get("higher_is_better", higher)),
                    review_reasons=review,
                )
                rows.append(row(c, verdict.decision, "", verdict, jv))
                self._remember(iteration, hypothesis, c, new_cols, verdict, base_scores, cand_scores)
            except Exception as exc:
                rows.append(row(c, "ERROR", type(exc).__name__, None, None, [f"{type(exc).__name__}: {exc}"]))
        return DiscoveryExperimentResult(report, pd.DataFrame(rows))

    def _remember(self, iteration, hypothesis, c, cols, verdict, base_scores, cand_scores):
        if self.memory is None:
            return
        self.memory.add(
            iteration=iteration,
            hypothesis=hypothesis or c.entity.description,
            candidate=c.entity.qualified_name,
            candidate_columns=cols,
            decision=verdict.decision,
            validation_strategy="runner_folds",
            join_path=c.join_plan.path if c.join_plan else [],
            baseline_scores=base_scores,
            candidate_scores=cand_scores,
            verdict=verdict,
        )
