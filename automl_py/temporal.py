from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureAvailability:
    feature: str
    available_offset: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    description: str = ""


class FeatureAvailabilityRegistry:
    """Declare when features are knowable relative to the prediction timestamp.

    Negative/zero offsets are available at prediction time. Positive offsets are future
    information and therefore leakage for a prediction made at time zero.
    """

    def __init__(self, mapping: Mapping[str, str | pd.Timedelta] | None = None):
        self._offsets: dict[str, pd.Timedelta] = {}
        for name, offset in (mapping or {}).items():
            self.register(name, offset)

    def register(self, feature: str, available_offset: str | pd.Timedelta = "0s"):
        self._offsets[feature] = pd.Timedelta(available_offset)
        return self

    def offset(self, feature: str) -> pd.Timedelta:
        return self._offsets.get(feature, pd.Timedelta(0))

    def unavailable(self, columns, prediction_horizon: str | pd.Timedelta = "0s") -> list[str]:
        # For a model made now, anything known after now is unavailable regardless of horizon.
        return [c for c in columns if self.offset(c) > pd.Timedelta(0)]

    def audit(self, columns) -> pd.DataFrame:
        rows = []
        for c in columns:
            off = self.offset(c)
            rows.append(
                {
                    "feature": c,
                    "available_offset": str(off),
                    "available_at_prediction": bool(off <= pd.Timedelta(0)),
                    "temporal_leakage_risk": bool(off > pd.Timedelta(0)),
                }
            )
        return pd.DataFrame(rows)

    def enforce(self, df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        audit = self.audit(df.columns)
        bad = audit.loc[audit.temporal_leakage_risk, "feature"].tolist()
        if bad and strict:
            raise ValueError("Features unavailable at prediction time: " + ", ".join(bad))
        return df.drop(columns=bad, errors="ignore"), audit
