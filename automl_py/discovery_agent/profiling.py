from __future__ import annotations
from dataclasses import dataclass
from .adapters.base import CatalogAdapter


@dataclass
class DiscoveryBudget:
    max_candidates_to_profile: int = 10
    max_sample_rows: int = 1000
    max_hops: int = 3


class BoundedProfiler:
    def __init__(self, adapter: CatalogAdapter, budget: DiscoveryBudget | None = None):
        self.adapter = adapter
        self.budget = budget or DiscoveryBudget()

    def profile(self, qualified_name: str) -> dict:
        return self.adapter.profile(qualified_name, sample_rows=self.budget.max_sample_rows)
