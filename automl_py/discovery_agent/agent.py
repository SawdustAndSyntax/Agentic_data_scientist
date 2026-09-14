from __future__ import annotations

import pandas as pd

from ..temporal import AVAILABLE, UNAVAILABLE, UNKNOWN, FeatureMetadata
from .adapters.base import CatalogAdapter
from .contracts import DatasetCandidate, DiscoveryRequest
from .graph import SemanticGraph
from .profiling import BoundedProfiler, DiscoveryBudget
from .scoring import HybridRelevanceScorer


def entity_availability(entity) -> tuple[str, str]:
    """Derive point-in-time availability from catalog metadata; UNKNOWN when undeclared.

    Recognised metadata keys: ``availability`` (available/unavailable/unknown),
    ``availability_offset`` (e.g. "-1d"), or ``event_time`` + ``available_time``.
    """
    md = entity.metadata or {}
    if md.get("availability") in {AVAILABLE, UNAVAILABLE, UNKNOWN}:
        return md["availability"], "declared in catalog metadata"
    if md.get("availability_offset") is not None or (md.get("event_time") and md.get("available_time")):
        meta = FeatureMetadata(
            entity.name,
            event_time=md.get("event_time"),
            available_time=md.get("available_time"),
            availability_offset=md.get("availability_offset"),
        )
        return meta.availability(), "derived from catalog availability offset"
    return UNKNOWN, "no availability metadata; requires review"


class DataDiscoveryAgent:
    """Search governed catalog/semantic metadata for testable predictive signal.

    Discovery is metadata-first and progressive: semantic metadata, then join
    reasoning, then bounded profiling of only the top candidates. The agent never
    creates/updates/deletes warehouse objects. Relevance identifies candidates; it
    never proves predictive value.
    """

    def __init__(self, adapter: CatalogAdapter, *, budget: DiscoveryBudget | None = None, scorer: HybridRelevanceScorer | None = None):
        self.adapter = adapter
        self.budget = budget or DiscoveryBudget()
        self.scorer = scorer or HybridRelevanceScorer()
        self.graph_: SemanticGraph | None = None

    def crawl(self) -> SemanticGraph:
        self.graph_ = SemanticGraph(self.adapter.entities(), self.adapter.relationships())
        return self.graph_

    def discover(
        self, request: DiscoveryRequest, *, anchor_entities: list[str] | None = None, profile: bool = True
    ) -> list[DatasetCandidate]:
        graph = self.graph_ or self.crawl()
        anchors = anchor_entities or []
        candidates = []
        for q, e in graph.entities.items():
            best_plan = None
            if anchors:
                plans = [graph.shortest_path(a, q, min(request.max_hops, self.budget.max_hops)) for a in anchors if a in graph.entities]
                plans = [p for p in plans if p]
                if plans:
                    best_plan = max(plans, key=lambda p: p.confidence / (1 + 0.15 * len(p.relationships)))
                joinability = best_plan.confidence / (1 + 0.10 * len(best_plan.relationships)) if best_plan else 0.0
            else:
                joinability = 0.5
            relevance, reasons = self.scorer.score(e, request, best_plan)
            if relevance <= 0:
                continue
            availability, why = entity_availability(e)
            temporal = {AVAILABLE: 1.0, UNAVAILABLE: 0.0}.get(availability)
            reasons.append(f"availability {availability} ({why})")
            c = DatasetCandidate(
                e,
                relevance,
                joinability,
                temporal,
                reasons=reasons,
                join_plan=best_plan,
                availability=availability,
                governance_status=str((e.metadata or {}).get("governance_status", "unknown")),
            )
            candidates.append(c)
        candidates.sort(key=lambda c: (c.score, c.relevance), reverse=True)
        if profile:
            profiler = BoundedProfiler(self.adapter, self.budget)
            for c in candidates[: self.budget.max_candidates_to_profile]:
                q = c.entity.metadata.get("base_table", c.entity.qualified_name)
                if str(q).startswith("semantic:"):
                    continue
                try:
                    p = profiler.profile(str(q))
                    c.coverage = float(p.get("coverage", c.coverage))
                    c.quality = float(p.get("quality", c.quality))
                except Exception as exc:
                    c.reasons.append(f"bounded profile unavailable ({type(exc).__name__}: {exc})")
            candidates.sort(key=lambda c: (c.score, c.relevance), reverse=True)
        return candidates[: request.top_k]

    @staticmethod
    def to_frame(candidates: list[DatasetCandidate]) -> pd.DataFrame:
        rows = []
        for c in candidates:
            rows.append(
                {
                    "candidate": c.entity.qualified_name,
                    "platform": c.entity.platform,
                    "object_type": c.entity.object_type,
                    "relevance": c.relevance,
                    "joinability": c.joinability,
                    "availability": c.availability,
                    "temporal_validity": c.temporal_validity,
                    "coverage": c.coverage,
                    "quality": c.quality,
                    "cost": c.cost,
                    "governance_status": c.governance_status,
                    "candidate_score": c.score,
                    "join_path": " -> ".join(c.join_plan.path) if c.join_plan else "",
                    "join_confidence": c.join_plan.confidence if c.join_plan else 0.0,
                    "reasons": " | ".join(c.reasons),
                }
            )
        return pd.DataFrame(rows)
