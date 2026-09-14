"""A final holdout that is locked until the champion is chosen.

The holdout is created before any model search, hidden from every experiment,
and evaluated exactly once. Repeated evaluation raises unless explicitly
forced, and every access attempt is recorded so an audit can prove the final
number was not used for selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .metrics import evaluate


class HoldoutAlreadyEvaluated(RuntimeError):
    """Raised when code attempts a second evaluation on the locked holdout."""


class HoldoutLocked(RuntimeError):
    """Raised when code tries to read holdout rows before the champion is locked."""


@dataclass(frozen=True)
class HoldoutEvaluation:
    target: str
    metric: str
    score: float
    n_rows: int
    model_tag: str
    evaluated_at: str


@dataclass
class FinalHoldout:
    target: str
    metric: str
    _X: pd.DataFrame = field(repr=False)
    _y: pd.Series = field(repr=False)
    evaluations: list[HoldoutEvaluation] = field(default_factory=list)
    access_log: list[dict] = field(default_factory=list)
    champion_locked: bool = False
    champion_tag: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self._y)

    @property
    def index(self) -> pd.Index:
        return self._y.index

    @property
    def evaluated(self) -> bool:
        return bool(self.evaluations)

    def lock_champion(self, tag: str):
        """Record that model search is finished and ``tag`` is the champion."""
        self.champion_locked = True
        self.champion_tag = tag
        self.access_log.append({"event": "champion_locked", "tag": tag, "at": _now()})
        return self

    def evaluate(self, model, tag: str = "champion", *, force: bool = False) -> HoldoutEvaluation:
        self.access_log.append({"event": "evaluate_requested", "tag": tag, "at": _now(), "force": force})
        if not self.champion_locked:
            self.access_log[-1]["outcome"] = "refused_unlocked"
            raise HoldoutLocked("Final holdout cannot be evaluated before lock_champion() is called")
        if self.evaluations and not force:
            self.access_log[-1]["outcome"] = "refused_repeat"
            raise HoldoutAlreadyEvaluated(
                f"Final holdout for '{self.target}' was already evaluated ({self.evaluations[0].model_tag}); "
                "the holdout is single-use. Candidate comparison must use development folds."
            )
        yp = model.predict(self._X)
        prob = None
        if self.metric == "roc_auc":
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(self._X)
            elif hasattr(model, "decision_function"):
                prob = model.decision_function(self._X)
        score = evaluate(self.metric, self._y, yp, prob)
        ev = HoldoutEvaluation(self.target, self.metric, float(score), self.n_rows, tag, _now())
        self.evaluations.append(ev)
        self.access_log[-1]["outcome"] = "evaluated" if len(self.evaluations) == 1 else "evaluated_forced_repeat"
        return ev

    def with_columns(self, frame: pd.DataFrame, columns=None):
        """Attach adopted candidate columns (index-aligned) to the holdout features.

        Only feature columns are added; the holdout target is never read. The event is logged.
        """
        cols = list(columns) if columns is not None else [c for c in frame.columns if c not in self._X.columns]
        self._X = self._X.join(frame[cols], how="left")
        self.access_log.append({"event": "columns_added", "columns": ",".join(cols), "at": _now()})
        return self

    def predictions(self, model) -> pd.DataFrame:
        """Return holdout predictions for reporting. Only allowed after the single evaluation."""
        if not self.evaluations:
            raise HoldoutLocked("Holdout predictions are only available after the one final evaluation")
        self.access_log.append({"event": "predictions_read", "at": _now()})
        return pd.DataFrame({"actual": np.asarray(self._y), "predicted": np.asarray(model.predict(self._X))}, index=self._y.index)

    def release(self, reason: str) -> tuple[pd.DataFrame, pd.Series]:
        """Expose the raw holdout rows. Recorded in the access log; use only after final evaluation."""
        self.access_log.append({"event": "released", "reason": reason, "at": _now(), "after_evaluation": self.evaluated})
        return self._X.copy(), self._y.copy()

    def audit(self) -> pd.DataFrame:
        return pd.DataFrame(self.access_log)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
