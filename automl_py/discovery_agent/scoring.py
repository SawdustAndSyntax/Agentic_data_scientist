"""Candidate relevance: a hybrid of deterministic and optional learned signals.

    lexical relevance + embedding similarity + graph distance + grain compatibility
    + joinability + historical experiment evidence -> candidate relevance

The lexical scorer is the guardrail: a candidate with zero lexical overlap and
no embedding support is never promoted. Relevance only identifies candidates;
predictive usefulness is established solely by the Experiment Judge.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import DiscoveryRequest, JoinPlan, SemanticEntity


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 1}


def semantic_relevance(entity: SemanticEntity, request: DiscoveryRequest) -> tuple[float, list[str]]:
    """Deterministic lexical relevance in [0, 1] with human-readable reasons."""
    query = tokens(" ".join([request.target, request.hypothesis, request.context, " ".join(request.grain)]))
    corpus = tokens(entity.searchable_text())
    overlap = query & corpus
    score = len(overlap) / max(1, len(query))
    # reward direct hypothesis match more strongly than generic context overlap
    h = tokens(request.hypothesis)
    h_overlap = h & corpus
    score = min(1.0, 0.45 * score + 0.55 * (len(h_overlap) / max(1, len(h))))
    reasons = []
    if h_overlap:
        reasons.append("hypothesis terms: " + ", ".join(sorted(h_overlap)))
    if overlap - h_overlap:
        reasons.append("context terms: " + ", ".join(sorted(overlap - h_overlap)[:8]))
    if entity.description:
        reasons.append("catalog description available")
    return score, reasons


@runtime_checkable
class EmbeddingRelevanceProvider(Protocol):
    """Optional dense-similarity provider (local model, hosted embeddings, ...)."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class HybridRelevanceScorer:
    """Combine lexical relevance with optional embedding, graph, grain and history signals.

    Weights sum to one over the signals that are actually available so that the
    deterministic core works with no optional provider installed.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingRelevanceProvider | None = None,
        memory=None,
        lexical_weight: float = 0.55,
        embedding_weight: float = 0.25,
        graph_weight: float = 0.10,
        grain_weight: float = 0.10,
        history_weight: float = 0.10,
        guardrail_min_embedding: float = 0.35,
    ):
        self.embedding_provider = embedding_provider
        self.memory = memory
        self.w = {"lexical": lexical_weight, "embedding": embedding_weight, "graph": graph_weight, "grain": grain_weight}
        self.history_weight = history_weight
        self.guardrail_min_embedding = guardrail_min_embedding
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray | None:
        if self.embedding_provider is None:
            return None
        if text not in self._embedding_cache:
            vec = np.asarray(next(iter(self.embedding_provider.embed([text]))), dtype=float)
            self._embedding_cache[text] = vec
        return self._embedding_cache[text]

    def embedding_similarity(self, entity: SemanticEntity, request: DiscoveryRequest) -> float | None:
        q = self._embed(" ".join([request.target, request.hypothesis, request.context]))
        e = self._embed(entity.searchable_text())
        if q is None or e is None or not q.size or not e.size:
            return None
        denom = np.linalg.norm(q) * np.linalg.norm(e)
        if denom == 0:
            return 0.0
        return float(max(0.0, min(1.0, (q @ e) / denom)))

    @staticmethod
    def grain_compatibility(entity: SemanticEntity, request: DiscoveryRequest) -> float | None:
        if not request.grain:
            return None
        names = {f.name.lower() for f in entity.fields} | {s.lower() for f in entity.fields for s in f.synonyms}
        hit = sum(1 for g in request.grain if g.lower() in names)
        return hit / len(request.grain)

    @staticmethod
    def graph_proximity(plan: JoinPlan | None) -> float | None:
        if plan is None:
            return None
        return float(plan.confidence / (1 + 0.20 * len(plan.relationships)))

    def score(self, entity: SemanticEntity, request: DiscoveryRequest, plan: JoinPlan | None = None) -> tuple[float, list[str]]:
        lexical, reasons = semantic_relevance(entity, request)
        signals = {"lexical": lexical}
        emb = self.embedding_similarity(entity, request)
        if emb is not None:
            signals["embedding"] = emb
            reasons.append(f"embedding similarity {emb:.2f}")
        graph = self.graph_proximity(plan)
        if graph is not None:
            signals["graph"] = graph
        grain = self.grain_compatibility(entity, request)
        if grain is not None:
            signals["grain"] = grain
            if grain > 0:
                reasons.append(f"grain compatibility {grain:.0%}")
        # guardrail: deterministic lexical relevance or strong embedding support is required
        if lexical <= 0 and (emb is None or emb < self.guardrail_min_embedding):
            return 0.0, reasons
        total_w = sum(self.w[k] for k in signals)
        score = sum(self.w[k] * v for k, v in signals.items()) / total_w
        if self.memory is not None:
            prior = self.memory.prior_for(entity.qualified_name)
            if prior:
                score += self.history_weight * prior
                reasons.append(f"historical experiment prior {prior:+.2f}")
        return float(max(0.0, min(1.0, score))), reasons
