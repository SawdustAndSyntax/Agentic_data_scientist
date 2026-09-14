"""Structured diagnostics so autonomous runs have observable failure states.

A failed diagnostic is never equivalent to a passed diagnostic: components
record a DiagnosticEvent instead of silently swallowing exceptions.
"""

from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import pandas as pd

log = logging.getLogger("automl_py")

Severity = str  # "INFO" | "WARNING" | "ERROR"


@dataclass(frozen=True)
class DiagnosticEvent:
    component: str
    severity: Severity
    message: str
    error_type: str | None = None
    experiment_id: str | None = None
    detail: str | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


class DiagnosticLog:
    def __init__(self, warn: bool = True):
        self.events: list[DiagnosticEvent] = []
        self.warn = warn

    def record(
        self,
        component: str,
        severity: Severity,
        message: str,
        *,
        error_type: str | None = None,
        experiment_id: str | None = None,
        detail: str | None = None,
    ) -> DiagnosticEvent:
        ev = DiagnosticEvent(component, severity, message, error_type, experiment_id, detail)
        self.events.append(ev)
        if severity == "ERROR":
            log.error("%s: %s", component, message)
            if self.warn:
                warnings.warn(f"[{component}] {message}", RuntimeWarning, stacklevel=3)
        elif severity == "WARNING":
            log.warning("%s: %s", component, message)
        else:
            log.info("%s: %s", component, message)
        return ev

    def error(self, component: str, exc: BaseException, *, experiment_id: str | None = None, context: str = "") -> DiagnosticEvent:
        msg = f"{context + ': ' if context else ''}{type(exc).__name__}: {exc}"
        return self.record(
            component,
            "ERROR",
            msg,
            error_type=type(exc).__name__,
            experiment_id=experiment_id,
            detail="".join(traceback.format_exception(exc, limit=3)),
        )

    @contextmanager
    def capture(self, component: str, *, experiment_id: str | None = None, context: str = "", reraise: bool = False) -> Iterator[None]:
        """Run a block; on failure record an ERROR event (and optionally re-raise)."""
        try:
            yield
        except Exception as exc:
            self.error(component, exc, experiment_id=experiment_id, context=context)
            if reraise:
                raise

    def errors(self) -> list[DiagnosticEvent]:
        return [e for e in self.events if e.severity == "ERROR"]

    @property
    def has_errors(self) -> bool:
        return any(e.severity == "ERROR" for e in self.events)

    def extend(self, other: DiagnosticLog):
        self.events.extend(other.events)
        return self

    def to_frame(self) -> pd.DataFrame:
        cols = ["at", "component", "severity", "message", "error_type", "experiment_id", "detail"]
        return pd.DataFrame([e.to_dict() for e in self.events], columns=cols)

    def __len__(self):
        return len(self.events)
