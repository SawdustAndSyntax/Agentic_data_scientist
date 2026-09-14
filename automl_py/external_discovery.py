from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import pandas as pd


@dataclass(frozen=True)
class ExternalDatasetCandidate:
    provider: str
    dataset_id: str
    name: str
    description: str
    concepts: tuple[str, ...] = ()
    geographic_grain: str | None = None
    temporal_grain: str | None = None
    historical_start: str | None = None
    historical_end: str | None = None
    annual_cost_estimate: float | None = None
    url: str | None = None


class ExternalCatalogProvider(Protocol):
    """
    Provider interface for Snowflake Marketplace, Databricks Marketplace,
    internal data exchanges, or approved third-party catalogs.

    Core AutoML-Py never silently purchases or ingests external data.
    """

    def search(
        self,
        query: str,
        concepts: Sequence[str] = (),
        limit: int = 20,
    ) -> list[ExternalDatasetCandidate]: ...


class ExternalSignalScout:
    def __init__(self, providers: Sequence[ExternalCatalogProvider]):
        self.providers = list(providers)

    def search(self, hypothesis: str, concepts: Sequence[str] = (), limit_per_provider: int = 10) -> pd.DataFrame:
        rows = []
        for provider in self.providers:
            try:
                candidates = provider.search(hypothesis, concepts=concepts, limit=limit_per_provider)
            except Exception as exc:
                rows.append(
                    {
                        "provider": type(provider).__name__,
                        "dataset_id": None,
                        "name": None,
                        "description": None,
                        "concepts": None,
                        "annual_cost_estimate": None,
                        "status": "provider_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for c in candidates:
                rows.append(
                    {
                        "provider": c.provider,
                        "dataset_id": c.dataset_id,
                        "name": c.name,
                        "description": c.description,
                        "concepts": ", ".join(c.concepts),
                        "geographic_grain": c.geographic_grain,
                        "temporal_grain": c.temporal_grain,
                        "historical_start": c.historical_start,
                        "historical_end": c.historical_end,
                        "annual_cost_estimate": c.annual_cost_estimate,
                        "url": c.url,
                        "status": "candidate",
                        "error": None,
                    }
                )
        return pd.DataFrame(rows)
