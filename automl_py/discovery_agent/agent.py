from __future__ import annotations
from dataclasses import asdict
import pandas as pd
from .contracts import DiscoveryRequest, DatasetCandidate
from .graph import SemanticGraph
from .profiling import BoundedProfiler, DiscoveryBudget
from .scoring import semantic_relevance
from .adapters.base import CatalogAdapter

class DataDiscoveryAgent:
    """Search governed catalog/semantic metadata for testable predictive signal.

    Discovery is metadata-first. Raw sampling is bounded and only performed for the
    highest-ranked candidates. The agent never creates/updates/deletes warehouse objects.
    """
    def __init__(self, adapter: CatalogAdapter, *, budget: DiscoveryBudget | None=None):
        self.adapter=adapter; self.budget=budget or DiscoveryBudget(); self.graph_: SemanticGraph | None=None

    def crawl(self) -> SemanticGraph:
        self.graph_=SemanticGraph(self.adapter.entities(),self.adapter.relationships())
        return self.graph_

    def discover(self, request: DiscoveryRequest, *, anchor_entities: list[str] | None=None, profile: bool=True) -> list[DatasetCandidate]:
        graph=self.graph_ or self.crawl(); anchors=anchor_entities or []
        candidates=[]
        for q,e in graph.entities.items():
            relevance,reasons=semantic_relevance(e,request)
            if relevance<=0: continue
            best_plan=None
            if anchors:
                plans=[graph.shortest_path(a,q,min(request.max_hops,self.budget.max_hops)) for a in anchors if a in graph.entities]
                plans=[p for p in plans if p]
                if plans: best_plan=max(plans,key=lambda p:p.confidence/(1+0.15*len(p.relationships)))
                joinability=best_plan.confidence/(1+0.10*len(best_plan.relationships)) if best_plan else 0.0
            else:
                joinability=0.5
            temporal=1.0  # actual feature-level as-of validation happens in the experiment scientist
            c=DatasetCandidate(e,relevance,joinability,temporal,reasons=reasons,join_plan=best_plan)
            candidates.append(c)
        candidates.sort(key=lambda c:(c.score,c.relevance),reverse=True)
        if profile:
            profiler=BoundedProfiler(self.adapter,self.budget)
            for c in candidates[:self.budget.max_candidates_to_profile]:
                q=c.entity.metadata.get('base_table',c.entity.qualified_name)
                if str(q).startswith('semantic:'): continue
                try:
                    p=profiler.profile(str(q)); c.coverage=float(p.get('coverage',c.coverage)); c.quality=float(p.get('quality',c.quality))
                except Exception:
                    c.reasons.append('bounded profile unavailable')
            candidates.sort(key=lambda c:(c.score,c.relevance),reverse=True)
        return candidates[:request.top_k]

    @staticmethod
    def to_frame(candidates: list[DatasetCandidate]) -> pd.DataFrame:
        rows=[]
        for c in candidates:
            rows.append({
                'candidate':c.entity.qualified_name,'platform':c.entity.platform,
                'object_type':c.entity.object_type,'relevance':c.relevance,
                'joinability':c.joinability,'temporal_validity':c.temporal_validity,
                'coverage':c.coverage,'quality':c.quality,'cost':c.cost,'candidate_score':c.score,
                'join_path':' -> '.join(c.join_plan.path) if c.join_plan else '',
                'join_confidence':c.join_plan.confidence if c.join_plan else 0.0,
                'reasons':' | '.join(c.reasons),
            })
        return pd.DataFrame(rows)
