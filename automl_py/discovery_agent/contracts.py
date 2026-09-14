from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

FieldRole = Literal['dimension','fact','metric','key','unknown']

@dataclass(frozen=True)
class SemanticField:
    name: str
    data_type: str = ''
    role: FieldRole = 'unknown'
    description: str = ''
    expression: str | None = None
    synonyms: tuple[str, ...] = ()

@dataclass
class SemanticEntity:
    name: str
    qualified_name: str
    platform: str
    object_type: str = 'table'
    description: str = ''
    fields: list[SemanticField] = field(default_factory=list)
    primary_keys: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        bits=[self.name,self.qualified_name,self.description]
        for f in self.fields:
            bits += [f.name,f.description,' '.join(f.synonyms)]
        return ' '.join(str(x) for x in bits if x).lower()

@dataclass(frozen=True)
class SemanticRelationship:
    name: str
    source: str
    target: str
    source_keys: tuple[str, ...] = ()
    target_keys: tuple[str, ...] = ()
    relationship_type: str = 'foreign_key'
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DiscoveryRequest:
    target: str
    hypothesis: str
    grain: tuple[str, ...] = ()
    context: str = ''
    prediction_horizon: str = '0s'
    max_hops: int = 3
    top_k: int = 10

@dataclass
class JoinPlan:
    path: list[str]
    relationships: list[SemanticRelationship]
    confidence: float
    warnings: list[str] = field(default_factory=list)

@dataclass
class DatasetCandidate:
    entity: SemanticEntity
    relevance: float
    joinability: float
    temporal_validity: float
    coverage: float = 0.5
    quality: float = 0.5
    cost: float = 0.0
    join_plan: JoinPlan | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        # Value-oriented and deliberately conservative: a candidate with no join
        # path or poor temporal validity cannot win on semantic relevance alone.
        positive=(
            0.30*self.relevance +
            0.25*self.joinability +
            0.20*self.temporal_validity +
            0.15*self.coverage +
            0.10*self.quality
        )
        return max(0.0, min(1.0, positive - 0.10*max(0.0,min(1.0,self.cost))))
