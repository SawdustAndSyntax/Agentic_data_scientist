"""Feature availability: was this feature knowable when the prediction was made?

This is a different control from the validation strategy (which asks whether the
model trained only on information that existed before each validation
observation). Both must pass.

Availability is one of three states. Unknown availability is never treated as
safe: it is surfaced as UNKNOWN so the experiment can be routed to REVIEW.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


def to_offset(value: str | pd.Timedelta) -> pd.Timedelta:
    """Parse an availability offset; accepts the short day suffix ('-1d') pandas is deprecating."""
    if isinstance(value, str):
        value = re.sub(r"(\d)\s*d$", r"\1D", value.strip())
    return pd.Timedelta(value)


AVAILABLE = "available"
UNAVAILABLE = "unavailable"
UNKNOWN = "unknown"
Availability = Literal["available", "unavailable", "unknown"]
UnknownPolicy = Literal["review", "strict", "allow"]


@dataclass(frozen=True)
class FeatureMetadata:
    """When a feature's value for an event becomes knowable.

    Either give ``event_time`` + ``available_time`` (absolute) or ``availability_offset``
    (relative to the prediction timestamp: negative/zero = knowable in advance,
    positive = only known afterwards). A feature with neither is UNKNOWN.
    """

    name: str
    event_time: str | pd.Timestamp | None = None
    available_time: str | pd.Timestamp | None = None
    source: str = ""
    availability_offset: str | pd.Timedelta | None = None
    description: str = ""

    def offset(self) -> pd.Timedelta | None:
        if self.availability_offset is not None:
            return to_offset(self.availability_offset)
        if self.event_time is not None and self.available_time is not None:
            return pd.Timestamp(self.available_time) - pd.Timestamp(self.event_time)
        return None

    def availability(self, prediction_time: str | pd.Timestamp | None = None) -> Availability:
        if prediction_time is not None and self.available_time is not None:
            return AVAILABLE if pd.Timestamp(self.available_time) <= pd.Timestamp(prediction_time) else UNAVAILABLE
        off = self.offset()
        if off is None:
            return UNKNOWN
        return AVAILABLE if off <= pd.Timedelta(0) else UNAVAILABLE


@dataclass(frozen=True)
class FeatureAvailability:
    """Backward-compatible offset-only record."""

    feature: str
    available_offset: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    description: str = ""


class FeatureAvailabilityRegistry:
    """Declare when features are knowable relative to the prediction timestamp.

    Negative/zero offsets are available at prediction time. Positive offsets are
    future information and therefore leakage. Features that were never registered
    have UNKNOWN availability.
    """

    def __init__(self, mapping: Mapping[str, str | pd.Timedelta | FeatureMetadata] | None = None):
        self._meta: dict[str, FeatureMetadata] = {}
        for name, value in (mapping or {}).items():
            if isinstance(value, FeatureMetadata):
                self.register_metadata(value)
            else:
                self.register(name, value)

    def register(self, feature: str, available_offset: str | pd.Timedelta = "0s", description: str = ""):
        self._meta[feature] = FeatureMetadata(feature, availability_offset=to_offset(available_offset), description=description)
        return self

    def register_metadata(self, meta: FeatureMetadata):
        self._meta[meta.name] = meta
        return self

    def metadata(self, feature: str) -> FeatureMetadata | None:
        return self._meta.get(feature)

    def known(self, feature: str) -> bool:
        return feature in self._meta and self._meta[feature].offset() is not None

    def offset(self, feature: str) -> pd.Timedelta | None:
        m = self._meta.get(feature)
        return None if m is None else m.offset()

    def status(self, feature: str, prediction_time=None) -> Availability:
        m = self._meta.get(feature)
        return UNKNOWN if m is None else m.availability(prediction_time)

    def unavailable(self, columns, prediction_time=None) -> list[str]:
        return [c for c in columns if self.status(c, prediction_time) == UNAVAILABLE]

    def unknown(self, columns, prediction_time=None) -> list[str]:
        return [c for c in columns if self.status(c, prediction_time) == UNKNOWN]

    def audit(self, columns, prediction_time=None) -> pd.DataFrame:
        rows = []
        for c in columns:
            status = self.status(c, prediction_time)
            off = self.offset(c)
            rows.append(
                {
                    "feature": c,
                    "availability": status,
                    "available_offset": None if off is None else str(off),
                    "available_at_prediction": None if status == UNKNOWN else status == AVAILABLE,
                    "temporal_leakage_risk": status == UNAVAILABLE,
                    "unknown_availability": status == UNKNOWN,
                    "source": (self._meta[c].source if c in self._meta else ""),
                }
            )
        cols = [
            "feature",
            "availability",
            "available_offset",
            "available_at_prediction",
            "temporal_leakage_risk",
            "unknown_availability",
            "source",
        ]
        return pd.DataFrame(rows, columns=cols)

    def enforce(
        self, df: pd.DataFrame, strict: bool = True, unknown_policy: UnknownPolicy = "review", prediction_time=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop unavailable features. ``unknown_policy``: review keeps and flags, strict raises, allow keeps silently."""
        audit = self.audit(df.columns, prediction_time)
        bad = audit.loc[audit.temporal_leakage_risk, "feature"].tolist()
        unknown = audit.loc[audit.unknown_availability, "feature"].tolist()
        if bad and strict:
            raise ValueError("Features unavailable at prediction time: " + ", ".join(bad))
        if unknown and unknown_policy == "strict":
            raise ValueError("Features with unknown availability (declare them in the registry): " + ", ".join(unknown))
        return df.drop(columns=bad, errors="ignore"), audit
