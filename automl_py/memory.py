"""Experiment memory: every experiment leaves an audit record, including failures.

Memory prevents the scientist from rerunning an identical failed experiment
without new evidence and accumulates organisational knowledge about which
information has mattered for which prediction problems.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .judge import INCONCLUSIVE, INVALID, KEEP, REJECT, REVIEW

TERMINAL_FAILURES = {REJECT, INVALID, INCONCLUSIVE}


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class ExperimentRecord:
    experiment_id: str
    iteration: int
    hypothesis_id: str | None
    hypothesis: str
    evidence: list[str]
    candidate: str
    candidate_columns: list[str]
    join_path: list[str]
    validation_strategy: str
    baseline_scores: list[float]
    candidate_scores: list[float]
    mean_uplift: float | None
    confidence_interval: list[float] | None
    positive_share: float | None
    noise_threshold: float | None
    required_gain: float | None
    decision: str
    reasons: list[str]
    fingerprint: str
    evidence_hash: str
    candidate_fingerprint: str = ""
    hypothesis_fingerprint: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def failed(self) -> bool:
        return self.decision in TERMINAL_FAILURES


class ExperimentMemory:
    def __init__(self, records: Sequence[ExperimentRecord] = ()):
        self.records: list[ExperimentRecord] = list(records)

    # -- identity -------------------------------------------------------------------------------
    def next_id(self) -> str:
        return f"EXP-{len(self.records) + 1:04d}"

    @staticmethod
    def fingerprint(candidate_columns: Sequence[str], validation_strategy: str, base_columns: Sequence[str] = ()) -> str:
        return stable_hash({"candidate": sorted(candidate_columns), "validation": validation_strategy, "base": sorted(base_columns)})

    @staticmethod
    def candidate_fingerprint(candidate_columns: Sequence[str]) -> str:
        return stable_hash(sorted(candidate_columns))

    @staticmethod
    def evidence_hash(evidence) -> str:
        return stable_hash(evidence)

    # -- recording ------------------------------------------------------------------------------
    def record(self, rec: ExperimentRecord) -> ExperimentRecord:
        self.records.append(rec)
        return rec

    def add(
        self,
        *,
        iteration: int,
        hypothesis,
        candidate: str,
        candidate_columns: Sequence[str],
        decision: str,
        validation_strategy: str,
        base_columns: Sequence[str] = (),
        join_path: Sequence[str] = (),
        baseline_scores: Sequence[float] = (),
        candidate_scores: Sequence[float] = (),
        verdict=None,
        reasons: Sequence[str] = (),
        extra: dict | None = None,
    ) -> ExperimentRecord:
        h_id = getattr(hypothesis, "hypothesis_id", None)
        h_stmt = getattr(hypothesis, "statement", str(hypothesis))
        evidence = list(getattr(hypothesis, "evidence", []))
        up = getattr(verdict, "uplift", None)
        rec = ExperimentRecord(
            experiment_id=self.next_id(),
            iteration=iteration,
            hypothesis_id=h_id,
            hypothesis=h_stmt,
            evidence=evidence,
            candidate=candidate,
            candidate_columns=list(candidate_columns),
            join_path=list(join_path),
            validation_strategy=validation_strategy,
            baseline_scores=[float(x) for x in baseline_scores],
            candidate_scores=[float(x) for x in candidate_scores],
            mean_uplift=None if up is None else float(up.mean),
            confidence_interval=None if up is None else [float(up.ci_low), float(up.ci_high)],
            positive_share=None if up is None else float(up.positive_share),
            noise_threshold=None if verdict is None else verdict.noise_threshold,
            required_gain=None if verdict is None else verdict.required_gain,
            decision=decision,
            reasons=list(reasons) or (list(verdict.reasons) if verdict is not None else []),
            fingerprint=self.fingerprint(candidate_columns, validation_strategy, base_columns),
            evidence_hash=self.evidence_hash(evidence),
            candidate_fingerprint=self.candidate_fingerprint(candidate_columns),
            hypothesis_fingerprint=str(getattr(hypothesis, "fingerprint", "") or ""),
            extra=dict(extra or {}),
        )
        return self.record(rec)

    # -- querying -------------------------------------------------------------------------------
    def find(self, fingerprint: str) -> list[ExperimentRecord]:
        return [r for r in self.records if r.fingerprint == fingerprint]

    def should_skip(self, fingerprint: str, evidence_hash: str | None = None, candidate_fingerprint: str | None = None) -> tuple[bool, str]:
        """Skip a repeat of an experiment already run under the same evidence.

        Returns (skip, reason). A previously KEPT candidate is skipped because it is
        already part of the feature set; a failed one is skipped unless the evidence
        hash differs (new evidence, changed data, or changed validation all alter it).
        An INVALID candidate (temporal leakage, governance, join failure) stays
        invalid whatever the baseline: it is skipped by candidate identity alone
        until a human re-registers its availability or provenance. A REJECTED or
        INCONCLUSIVE candidate is likewise not re-tested against a merely different
        baseline unless the hypothesis evidence has changed.
        """
        if candidate_fingerprint:
            prior_c = [r for r in self.records if r.candidate_fingerprint == candidate_fingerprint]
            invalid = [r for r in prior_c if r.decision == INVALID]
            if invalid:
                return True, f"{INVALID} in {invalid[-1].experiment_id}: {'; '.join(invalid[-1].reasons)[:120]}"
            if prior_c:
                last = prior_c[-1]
                if last.decision in {KEEP, REVIEW}:
                    return True, f"already {last.decision} in {last.experiment_id}"
                if last.failed and (evidence_hash is None or last.evidence_hash == evidence_hash):
                    return True, f"{last.decision} in {last.experiment_id} with identical evidence"
        prior = self.find(fingerprint)
        if not prior:
            return False, ""
        # Same candidate, same folds, same baseline columns: the outcome is deterministic,
        # so re-running it under a different hypothesis cannot add information.
        last = prior[-1]
        return True, f"identical experiment already run ({last.experiment_id}: {last.decision})"

    def should_skip_hypothesis(self, hypothesis_fingerprint: str, evidence_hash: str) -> tuple[bool, str]:
        """A hypothesis whose experiments all failed under identical evidence is not re-planned."""
        prior = [r for r in self.records if r.hypothesis_fingerprint == hypothesis_fingerprint]
        if not prior:
            return False, ""
        if any(r.decision in {KEEP, REVIEW} for r in prior):
            return False, ""
        same_evidence = [r for r in prior if r.evidence_hash == evidence_hash]
        if same_evidence and all(r.failed for r in same_evidence):
            return (
                True,
                f"all {len(same_evidence)} experiment(s) for this hypothesis failed under identical evidence ({same_evidence[-1].experiment_id})",
            )
        return False, ""

    def by_decision(self, decision: str) -> list[ExperimentRecord]:
        return [r for r in self.records if r.decision == decision]

    def kept(self) -> list[ExperimentRecord]:
        return self.by_decision(KEEP)

    def decisions(self) -> dict[str, str]:
        """Latest decision per candidate name."""
        out: dict[str, str] = {}
        for r in self.records:
            out[r.candidate] = r.decision
        return out

    def prior_for(self, candidate: str) -> float:
        """Historical evidence prior in [-1, 1] for relevance scoring."""
        decs = [r.decision for r in self.records if r.candidate == candidate]
        if not decs:
            return 0.0
        score = 0.0
        for d in decs:
            score += {KEEP: 0.5, REVIEW: 0.2, INCONCLUSIVE: -0.1, REJECT: -0.4, INVALID: -0.5}.get(d, 0.0)
        return max(-1.0, min(1.0, score))

    # -- persistence / reporting -----------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        cols = [f.name for f in ExperimentRecord.__dataclass_fields__.values()]
        return pd.DataFrame([r.to_dict() for r in self.records], columns=cols)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([r.to_dict() for r in self.records], indent=2, default=str))
        return path

    @classmethod
    def load(cls, path) -> ExperimentMemory:
        data = json.loads(Path(path).read_text())
        return cls([ExperimentRecord(**d) for d in data])

    def summary(self) -> str:
        if not self.records:
            return "No experiments recorded."
        lines = ["Experiment memory", "=" * 17]
        for r in self.records:
            up = "" if r.mean_uplift is None else f" uplift={r.mean_uplift:+.4f}"
            lines.append(f"{r.experiment_id} it{r.iteration} {r.candidate:<28} {r.decision:<12}{up}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.records)
