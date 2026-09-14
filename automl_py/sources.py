"""Candidate sources for the orchestrator.

* ``DerivedFeatureSource`` turns *derivable* hypotheses ("residual still depends on
  x") into engineered candidates: nonlinear transforms, interactions, and for
  temporal problems lags and rolling statistics that only use past values.
* ``DiscoveryCandidateSource`` bridges the governed ``DataDiscoveryAgent`` into the
  loop: it discovers catalog candidates for a hypothesis, loads them through an
  injected loader, joins them point-in-time to the development frame, and hands
  the orchestrator index-aligned candidate columns with availability and join
  provenance. Row explosion and low coverage surface as INVALID at the gate.
* ``PointInTimeJoiner`` performs as-of joins that respect an availability lag: a
  value observed at time t is only usable at t + lag.
* ``CompositeCandidateSource`` combines sources.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .discovery_agent.agent import DataDiscoveryAgent
from .discovery_agent.contracts import DatasetCandidate, DiscoveryRequest
from .hypotheses import Hypothesis
from .orchestrator import CandidateFeatureSet, LoopState
from .temporal import AVAILABLE, FeatureAvailabilityRegistry


class CompositeCandidateSource:
    def __init__(self, sources: Sequence):
        self.sources = list(sources)

    def candidates(self, hypothesis: Hypothesis, state: LoopState) -> list[CandidateFeatureSet]:
        out, seen = [], set()
        for src in self.sources:
            for c in src.candidates(hypothesis, state):
                if c.name not in seen:
                    seen.add(c.name)
                    out.append(c)
        return out


# --------------------------------------------------------------------------------------------- #
# Derived features
# --------------------------------------------------------------------------------------------- #
class DerivedFeatureSource:
    """Engineer candidates from a feature the residual still depends on.

    Derived columns inherit availability from their inputs: every input already
    passed the availability audit to be in the model, and lags/rolling statistics
    use ``shift >= 1`` so they never touch the current or future period.
    """

    def __init__(
        self, *, max_interactions: int = 2, lags: Sequence[int] = (1,), rolling_windows: Sequence[int] = (3,), transforms: bool = True
    ):
        self.max_interactions = max_interactions
        self.lags = tuple(lags)
        self.rolling_windows = tuple(rolling_windows)
        self.transforms = transforms

    def candidates(self, hypothesis: Hypothesis, state: LoopState) -> list[CandidateFeatureSet]:
        if hypothesis.kind != "derivable":
            return []
        f = hypothesis.signals.get("feature")
        if not f or f not in state.X.columns or not pd.api.types.is_numeric_dtype(state.X[f]):
            return []
        col = state.X[f].astype(float)
        out: list[CandidateFeatureSet] = []
        if self.transforms:
            frame = pd.DataFrame({f"{f}__squared": col**2, f"{f}__log1p_abs": np.log1p(col.abs())}, index=state.X.index)
            out.append(self._make(f"derived:{f}:nonlinear", frame, (f, "nonlinear"), [f]))
        others = [c for c in state.X.columns if c != f and pd.api.types.is_numeric_dtype(state.X[c]) and state.X[c].nunique() > 2]
        if others and self.max_interactions:
            redundancy = {
                c: abs(float(np.corrcoef(col.fillna(col.mean()), state.X[c].astype(float).fillna(state.X[c].mean()))[0, 1])) for c in others
            }
            for g in sorted(others, key=lambda c: redundancy[c])[: self.max_interactions]:
                frame = pd.DataFrame({f"{f}__x__{g}": col * state.X[g].astype(float)}, index=state.X.index)
                out.append(self._make(f"derived:{f}x{g}", frame, (f, g, "interaction"), [f, g]))
        if state.timestamps is not None and (self.lags or self.rolling_windows):
            work = pd.DataFrame({"_v": col.to_numpy(), "_t": np.asarray(state.timestamps)}, index=state.X.index)
            work["_g"] = np.asarray(state.groups) if state.groups is not None else 0
            work = work.sort_values(["_g", "_t"])
            grouped = work.groupby("_g")["_v"]
            cols = {}
            for k in self.lags:
                cols[f"{f}__lag{k}"] = grouped.shift(k)
            for w in self.rolling_windows:
                cols[f"{f}__rolling{w}_mean"] = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            frame = pd.DataFrame(cols).reindex(state.X.index)
            out.append(self._make(f"derived:{f}:history", frame, (f, "lag", "rolling"), [f]))
        return out

    @staticmethod
    def _make(name, frame, concepts, inputs) -> CandidateFeatureSet:
        return CandidateFeatureSet(
            name,
            frame,
            tuple(concepts),
            availability=dict.fromkeys(frame.columns, AVAILABLE),
            source="derived",
            join_path=[f"derived from {', '.join(inputs)}"],
        )


# --------------------------------------------------------------------------------------------- #
# Point-in-time joins
# --------------------------------------------------------------------------------------------- #
@dataclass
class PointInTimeJoiner:
    """As-of join: for each left row take the latest right row whose (lagged) time is <= left time.

    ``availability_lag`` shifts the right-hand timestamps forward so a value
    observed at t becomes usable only at t + lag. ``tolerance`` bounds how stale a
    match may be. Times may be datetimes or integer periods (same dtype on both sides).
    """

    tolerance: object | None = None
    allow_exact_matches: bool = True

    def join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        left_time: str,
        right_time: str,
        left_keys: Sequence[str] = (),
        right_keys: Sequence[str] = (),
        availability_lag=None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        r = right.copy()
        if availability_lag is not None:
            r[right_time] = r[right_time] + availability_lag
        value_cols = list(columns) if columns is not None else [c for c in r.columns if c not in {right_time, *right_keys}]
        keep = [right_time, *right_keys, *value_cols]
        r = r[keep].sort_values(right_time)
        left_keys, right_keys = list(left_keys), list(right_keys)
        lft = left[[left_time, *left_keys]].copy()
        lft["__row"] = np.arange(len(lft))
        lft = lft.sort_values(left_time)
        kwargs = {}
        if left_keys:
            if left_keys == right_keys:
                kwargs["by"] = left_keys
            else:
                kwargs["left_by"], kwargs["right_by"] = left_keys, right_keys
        merged = pd.merge_asof(
            lft,
            r,
            left_on=left_time,
            right_on=right_time,
            direction="backward",
            tolerance=self.tolerance,
            allow_exact_matches=self.allow_exact_matches,
            suffixes=("", "__candidate"),
            **kwargs,
        )
        merged = merged.sort_values("__row")
        out = merged[value_cols].copy()
        out.index = left.index
        return out


def key_join(left: pd.DataFrame, right: pd.DataFrame, left_keys: Sequence[str], right_keys: Sequence[str], columns=None) -> pd.DataFrame:
    """Plain key join returning right-hand columns aligned to ``left.index``.

    A one-to-many join yields a duplicated index, which the orchestrator gate reports
    as ROW_EXPLOSION before any training happens.
    """
    left_keys, right_keys = list(left_keys), list(right_keys)
    value_cols = list(columns) if columns is not None else [c for c in right.columns if c not in right_keys]
    lft = left[left_keys].copy()
    lft["__idx"] = left.index
    merged = lft.merge(right[[*right_keys, *value_cols]], left_on=left_keys, right_on=right_keys, how="left", suffixes=("", "__candidate"))
    out = merged[value_cols].copy()
    out.index = pd.Index(merged["__idx"])
    return out


# --------------------------------------------------------------------------------------------- #
# Discovery-backed source
# --------------------------------------------------------------------------------------------- #
@dataclass
class CandidateData:
    """What a loader returns for a discovered candidate."""

    frame: pd.DataFrame
    left_keys: Sequence[str] = ()
    right_keys: Sequence[str] = ()
    right_time: str | None = None  # column in ``frame`` holding the observation time (enables as-of join)
    availability_lag: object | None = None  # e.g. pd.Timedelta("1D") or 1 period
    columns: Sequence[str] | None = None
    cost: float = 0.0
    queries: int = 1
    availability: dict[str, str] | None = None  # per-column override; else the catalog candidate's availability


class DiscoveryCandidateSource:
    """Feed governed catalog candidates into the orchestrator.

    ``loader(candidate) -> CandidateData | None`` performs the (bounded, approved)
    extraction. Time-keyed data is joined as-of with the candidate's availability
    lag; key-only data is joined on keys. Candidates whose catalog availability is
    ``unavailable`` are still returned so the gate records an INVALID experiment
    with its reason rather than silently dropping them.
    """

    def __init__(
        self,
        agent: DataDiscoveryAgent,
        loader: Callable[[DatasetCandidate], CandidateData | None],
        *,
        anchor_entities: Sequence[str] = (),
        timestamp_column: str | None = None,
        request_context: str = "",
        grain: Sequence[str] = (),
        top_k: int = 5,
        profile: bool = True,
        joiner: PointInTimeJoiner | None = None,
        feature_availability: FeatureAvailabilityRegistry | None = None,
    ):
        self.agent = agent
        self.loader = loader
        self.anchor_entities = list(anchor_entities)
        self.timestamp_column = timestamp_column
        self.request_context = request_context
        self.grain = tuple(grain)
        self.top_k = top_k
        self.profile = profile
        self.joiner = joiner or PointInTimeJoiner()
        self.feature_availability = feature_availability
        self.last_report: pd.DataFrame | None = None

    def candidates(self, hypothesis: Hypothesis, state: LoopState) -> list[CandidateFeatureSet]:
        if hypothesis.kind == "unknown" or not hypothesis.search_concepts or state.frame is None:
            return []
        request = DiscoveryRequest(
            target=state.target,
            hypothesis=" ".join(hypothesis.search_concepts),
            context=self.request_context or hypothesis.statement,
            grain=self.grain,
            top_k=self.top_k,
        )
        found = self.agent.discover(request, anchor_entities=self.anchor_entities, profile=self.profile)
        self.last_report = self.agent.to_frame(found)
        out: list[CandidateFeatureSet] = []
        for cand in found:
            data = self.loader(cand)
            if data is None or data.frame is None or data.frame.empty:
                continue
            base = state.frame
            if data.right_time and self.timestamp_column:
                aligned = self.joiner.join(
                    base,
                    data.frame,
                    left_time=self.timestamp_column,
                    right_time=data.right_time,
                    left_keys=data.left_keys,
                    right_keys=data.right_keys,
                    availability_lag=data.availability_lag,
                    columns=data.columns,
                )
            else:
                aligned = key_join(base, data.frame, data.left_keys, data.right_keys, data.columns)
            cols = [c for c in aligned.columns if c not in state.feature_columns]
            if not cols:
                continue
            availability = data.availability or {
                c: (self.feature_availability.status(c) if self.feature_availability else cand.availability) for c in cols
            }
            out.append(
                CandidateFeatureSet(
                    cand.entity.qualified_name,
                    aligned[cols],
                    tuple(hypothesis.search_concepts),
                    columns=cols,
                    join_path=list(cand.join_plan.path) if cand.join_plan else [cand.entity.qualified_name],
                    availability=availability,
                    cost=data.cost,
                    governance_status=cand.governance_status if cand.governance_status not in {"unknown", ""} else "approved",
                    source=f"discovery:{cand.entity.platform}",
                    queries=data.queries,
                )
            )
        return out


__all__ = [
    "CandidateData",
    "CompositeCandidateSource",
    "DerivedFeatureSource",
    "DiscoveryCandidateSource",
    "PointInTimeJoiner",
    "field",
    "key_join",
]
