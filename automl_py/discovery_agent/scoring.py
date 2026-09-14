from __future__ import annotations

import re

from .contracts import DiscoveryRequest, SemanticEntity


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 1}


def semantic_relevance(entity: SemanticEntity, request: DiscoveryRequest) -> tuple[float, list[str]]:
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
