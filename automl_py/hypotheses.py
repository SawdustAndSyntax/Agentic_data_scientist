"""Evidence-backed hypothesis generation.

Hypotheses are derived from the diagnosis of the current champion (residual
patterns by period and segment, residual/feature association, missingness,
drift, absent concept families) and filtered against experiment memory. A
hypothesis always records why it exists. When residual structure suggests
missing information but no defensible hypothesis can be formed, the generator
returns an explicit UNKNOWN_SIGNAL hypothesis rather than fabricating one.

``HypothesisReasoner`` is an optional hook (for example an LLM) that may add or
re-rank hypotheses; the deterministic generator never depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .discovery import LIB
from .memory import ExperimentMemory, stable_hash

HypothesisKind = str  # "existing" | "derivable" | "external" | "unknown"
UNKNOWN_SIGNAL = "UNKNOWN_SIGNAL"


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    evidence: list[str]
    search_concepts: list[str]
    confidence: float
    kind: HypothesisKind = "external"
    concept: str = ""
    signals: dict = field(default_factory=dict)

    @property
    def evidence_hash(self) -> str:
        return stable_hash(self.evidence)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"concept": self.concept, "kind": self.kind, "concepts": sorted(self.search_concepts)})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    """Everything the generator needs about the current state of the model."""

    target: str
    metric: str
    baseline_scores: list[float]
    current_columns: list[str]
    residual_frame: pd.DataFrame | None = None  # columns: actual, predicted, residual (+ features)
    residual_diagnostics: pd.DataFrame | None = None  # feature <-> residual association
    missingness: pd.DataFrame | None = None
    leakage: pd.DataFrame | None = None
    drift: dict | None = None
    timestamps: pd.Series | None = None
    groups: pd.Series | None = None
    context: str = ""
    feature_opportunities: pd.DataFrame | None = None


@runtime_checkable
class HypothesisReasoner(Protocol):
    """Optional refinement hook: receives the deterministic hypotheses and the diagnosis."""

    def refine(self, hypotheses: list[Hypothesis], diagnosis: Diagnosis) -> list[Hypothesis]: ...


class HypothesisGenerator:
    def __init__(
        self,
        *,
        library: dict | None = None,
        reasoner: HypothesisReasoner | None = None,
        max_hypotheses: int = 10,
        period_error_ratio: float = 1.20,
        segment_error_ratio: float = 1.25,
        residual_mi_threshold: float = 0.02,
    ):
        self.library = LIB if library is None else library
        self.reasoner = reasoner
        self.max_hypotheses = max_hypotheses
        self.period_error_ratio = period_error_ratio
        self.segment_error_ratio = segment_error_ratio
        self.residual_mi_threshold = residual_mi_threshold
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"H-{self._counter:03d}"

    # ------------------------------------------------------------------------------------------ #
    def generate(self, diagnosis: Diagnosis, memory: ExperimentMemory | None = None) -> list[Hypothesis]:
        hyps: list[Hypothesis] = []
        cols_text = " ".join(map(str, diagnosis.current_columns)).lower()
        residual_structure = False

        # 1. temporal error pattern ------------------------------------------------------------
        period_evidence = self._period_pattern(diagnosis)
        if period_evidence:
            residual_structure = True

        # 2. segment error pattern --------------------------------------------------------------
        segment_evidence, worst_group = self._segment_pattern(diagnosis)
        if segment_evidence:
            residual_structure = True
            hyps.append(
                Hypothesis(
                    self._new_id(),
                    f"Segment-level attributes may explain concentrated error in group '{worst_group}'.",
                    segment_evidence,
                    ["segment attributes", "profile", "location", "category", str(worst_group)],
                    0.55,
                    kind="existing",
                    concept="segment_attributes",
                    signals={"worst_group": str(worst_group)},
                )
            )

        # 3. residual / feature association -> derivable signal -----------------------------------
        rd = diagnosis.residual_diagnostics
        if rd is not None and not rd.empty:
            top = rd.iloc[0]
            mi = float(top.get("residual_mutual_information", np.nan))
            corr = float(top.get("residual_correlation", np.nan))
            if np.isfinite(mi) and mi >= self.residual_mi_threshold:
                residual_structure = True
                hyps.append(
                    Hypothesis(
                        self._new_id(),
                        f"Residual error still depends on '{top.feature}'; a nonlinear transform or interaction may recover signal.",
                        [f"residual mutual information with {top.feature} = {mi:.3f}", f"residual correlation = {corr:+.3f}"],
                        [str(top.feature), f"{top.feature} interaction", f"{top.feature} nonlinear"],
                        min(0.8, 0.4 + mi),
                        kind="derivable",
                        concept=f"residual:{top.feature}",
                        signals={"feature": str(top.feature), "mutual_information": mi},
                    )
                )

        # 4. systematic missingness ---------------------------------------------------------------
        ms = diagnosis.missingness
        if ms is not None and not ms.empty and "missingness_predictability_auc" in ms:
            sys_ = ms[ms.missingness_predictability_auc.fillna(0) >= 0.65]
            if not sys_.empty:
                feats = ", ".join(sys_.feature.head(3))
                hyps.append(
                    Hypothesis(
                        self._new_id(),
                        f"Missingness in {feats} is systematic and may itself carry signal or indicate an upstream process variable.",
                        [
                            f"missingness of {r.feature} predictable with AUC {r.missingness_predictability_auc:.2f}"
                            for r in sys_.head(3).itertuples()
                        ],
                        ["missingness", "data capture process", *sys_.feature.head(3).tolist()],
                        0.5,
                        kind="derivable",
                        concept="missingness",
                    )
                )

        # 5. drift -----------------------------------------------------------------------------------
        drift = diagnosis.drift
        if drift and float(drift.get("auc", 0.5)) >= 0.65:
            top_drift = ", ".join(drift["feature_importance"].feature.head(3)) if "feature_importance" in drift else ""
            hyps.append(
                Hypothesis(
                    self._new_id(),
                    "The evaluated periods differ from training; regime or trend information may be missing.",
                    [f"adversarial validation AUC {float(drift['auc']):.3f}", f"drivers: {top_drift}"],
                    ["trend", "regime", "seasonality", "macro"],
                    0.45,
                    kind="existing",
                    concept="drift",
                )
            )

        # 6. absent concept families -----------------------------------------------------------------
        text = f"{diagnosis.target} {diagnosis.context}".lower()
        demand = any(k in text for k in ["sales", "demand", "revenue", "volume", "traffic", "orders"])
        for concept, (terms, why, grain) in self.library.items():
            if any(t in cols_text for t in terms):
                continue
            base_conf = (
                0.6
                if (
                    demand and concept in {"weather", "calendar", "price_promotion", "inventory_availability", "events", "mobility_traffic"}
                )
                else 0.35
            )
            evidence = [f"no {concept} variables in current feature set", why]
            if period_evidence and concept in {"weather", "calendar", "events"}:
                evidence = period_evidence + evidence
                base_conf += 0.15
            hyps.append(
                Hypothesis(
                    self._new_id(),
                    f"{concept.replace('_', ' ').capitalize()} information may explain residual variation in {diagnosis.target}.",
                    evidence,
                    list(terms),
                    min(0.9, base_conf),
                    kind="external",
                    concept=concept,
                    signals={"grain": grain},
                )
            )

        # 7. unknown signal --------------------------------------------------------------------------
        if not hyps or (residual_structure and all(h.kind == "external" and h.confidence < 0.5 for h in hyps)):
            hyps.append(
                Hypothesis(
                    self._new_id(),
                    "Residual structure suggests missing information but no defensible candidate concept was identified.",
                    (period_evidence or segment_evidence or ["no strong residual/feature association detected"]),
                    [],
                    0.2,
                    kind="unknown",
                    concept=UNKNOWN_SIGNAL,
                )
            )

        if self.reasoner is not None:
            hyps = list(self.reasoner.refine(hyps, diagnosis))

        hyps = self._filter_memory(hyps, memory)
        hyps.sort(key=lambda h: h.confidence, reverse=True)
        return hyps[: self.max_hypotheses]

    # ------------------------------------------------------------------------------------------ #
    def _filter_memory(self, hyps: list[Hypothesis], memory: ExperimentMemory | None) -> list[Hypothesis]:
        if memory is None:
            return hyps
        out = []
        for h in hyps:
            skip, _ = memory.should_skip_hypothesis(h.fingerprint, h.evidence_hash)
            if not skip:
                out.append(h)
        return out

    def _period_pattern(self, d: Diagnosis) -> list[str]:
        rf, ts = d.residual_frame, d.timestamps
        if rf is None or ts is None or rf.empty or "residual" not in rf:
            return []
        ts = pd.Series(np.asarray(ts), index=rf.index) if len(ts) == len(rf) else None
        if ts is None:
            return []
        err = rf.residual.abs()
        if pd.api.types.is_datetime64_any_dtype(ts):
            bucket = ts.dt.month.rename("period")
            label = "month"
        else:
            bucket = pd.qcut(ts.rank(method="first"), q=min(4, ts.nunique()), labels=False, duplicates="drop").rename("period")
            label = "period quartile"
        by = err.groupby(bucket).mean()
        overall = float(err.mean())
        if overall <= 0 or by.empty:
            return []
        worst = by.idxmax()
        ratio = float(by.max() / overall)
        if ratio >= self.period_error_ratio:
            return [f"mean absolute error in {label} {worst} is {ratio - 1:.0%} above the overall average"]
        return []

    def _segment_pattern(self, d: Diagnosis) -> tuple[list[str], object]:
        rf, gr = d.residual_frame, d.groups
        if rf is None or gr is None or rf.empty or "residual" not in rf or len(gr) != len(rf):
            return [], None
        err = rf.residual.abs()
        by = err.groupby(np.asarray(gr)).mean()
        overall = float(err.mean())
        if overall <= 0 or len(by) < 2:
            return [], None
        worst = by.idxmax()
        ratio = float(by.max() / overall)
        if ratio >= self.segment_error_ratio:
            return [f"mean absolute error for segment '{worst}' is {ratio - 1:.0%} above the overall average"], worst
        return [], None


def hypotheses_frame(hyps: Sequence[Hypothesis]) -> pd.DataFrame:
    cols = ["hypothesis_id", "kind", "concept", "confidence", "statement", "evidence", "search_concepts"]
    rows = [{**h.to_dict(), "evidence": " | ".join(h.evidence), "search_concepts": ", ".join(h.search_concepts)} for h in hyps]
    return pd.DataFrame(rows, columns=cols)
