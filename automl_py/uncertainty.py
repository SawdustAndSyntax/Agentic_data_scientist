from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


@dataclass
class ConformalPrediction:
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    radius: float


class SplitConformalRegressor:
    """Simple distribution-free split-conformal intervals for regression."""

    def __init__(self, estimator, alpha: float = 0.10, calibration_size: float = 0.2, random_state: int = 100, chronological: bool = False):
        """``chronological=True`` reserves the *last* ``calibration_size`` fraction of rows (already
        sorted by time) for calibration instead of a random sample; use it for temporal problems."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.chronological = chronological

    def fit(self, X: pd.DataFrame, y):
        if self.chronological:
            n_cal = max(1, round(len(X) * self.calibration_size))
            Xfit, Xcal = X.iloc[:-n_cal], X.iloc[-n_cal:]
            yfit, ycal = pd.Series(np.asarray(y)).iloc[:-n_cal], pd.Series(np.asarray(y)).iloc[-n_cal:]
        else:
            Xfit, Xcal, yfit, ycal = train_test_split(X, y, test_size=self.calibration_size, random_state=self.random_state)
        self.model_ = clone(self.estimator)
        self.model_.fit(Xfit, yfit)
        pred = self.model_.predict(Xcal)
        resid = np.abs(np.asarray(ycal) - np.asarray(pred))
        n = len(resid)
        q = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        try:
            self.radius_ = float(np.quantile(resid, q, method="higher"))
        except TypeError:
            self.radius_ = float(np.quantile(resid, q, interpolation="higher"))
        return self

    def predict(self, X: pd.DataFrame) -> ConformalPrediction:
        p = np.asarray(self.model_.predict(X))
        return ConformalPrediction(p, p - self.radius_, p + self.radius_, self.alpha, self.radius_)

    def empirical_coverage(self, X: pd.DataFrame, y) -> float:
        p = self.predict(X)
        yy = np.asarray(y)
        return float(np.mean((yy >= p.lower) & (yy <= p.upper)))
