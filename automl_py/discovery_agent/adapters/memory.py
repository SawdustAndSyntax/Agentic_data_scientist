from __future__ import annotations

from .base import CatalogAdapter


class InMemoryCatalogAdapter(CatalogAdapter):
    platform = "memory"

    def __init__(self, entities, relationships=None, frames=None):
        self._entities = list(entities)
        self._relationships = list(relationships or [])
        self.frames = frames or {}

    def entities(self):
        return list(self._entities)

    def relationships(self):
        return list(self._relationships)

    def sample(self, qualified_name, limit=1000):
        if qualified_name not in self.frames:
            raise KeyError(qualified_name)
        return self.frames[qualified_name].head(limit).copy()

    def profile(self, qualified_name, sample_rows=1000):
        df = self.sample(qualified_name, sample_rows)
        if df.empty:
            return {"qualified_name": qualified_name, "sample_rows": 0, "coverage": 0.0, "quality": 0.0}
        coverage = float(1 - df.isna().mean().mean())
        quality = float(max(0.0, min(1.0, coverage)))
        return {"qualified_name": qualified_name, "sample_rows": len(df), "coverage": coverage, "quality": quality}
