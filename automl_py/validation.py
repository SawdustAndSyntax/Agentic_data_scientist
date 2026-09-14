"""First-class validation strategies.

Every experiment in Predictive Discovery (model search, tuning, feature-family
experiments, candidate-dataset experiments, noise controls, stability and
conformal calibration) must share one materialized set of folds so that
baseline and candidate are compared under identical conditions.

Temporal strategies never place an observation in a training fold whose
timestamp is on or after the earliest timestamp of that fold's validation
window.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold as _SkGroupKFold
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, train_test_split

StrategyName = Literal["random_kfold", "stratified_kfold", "group_kfold", "rolling_origin", "expanding_window", "sliding_window"]
TEMPORAL_STRATEGIES = {"rolling_origin", "expanding_window", "sliding_window"}

Fold = tuple[np.ndarray, np.ndarray]


@runtime_checkable
class ValidationStrategy(Protocol):
    """A splitter that yields positional (train, validation) index arrays."""

    @property
    def strategy_name(self) -> str: ...

    def split(self, X, y=None, groups=None, timestamps=None) -> Iterator[Fold]: ...


@dataclass(slots=True)
class ValidationConfig:
    """Declarative description of how experiments are validated.

    ``timestamp_column`` is required for temporal strategies and ``group_columns``
    for ``group_kfold``. For grouped temporal problems (store x week, SKU x week)
    use a temporal strategy and pass ``group_columns``: every split is made on the
    period axis so all groups move forward in time together.
    """

    strategy: StrategyName = "random_kfold"
    timestamp_column: str | None = None
    group_columns: Sequence[str] = ()
    horizon: int = 1
    gap: int = 0
    min_train_periods: int | None = None
    test_periods: int = 1
    step: int | None = None
    train_periods: int | None = None
    max_splits: int | None = None
    n_splits: int = 5
    n_repeats: int = 1
    random_state: int = 100
    holdout_fraction: float | None = None
    holdout_periods: int | None = None

    @property
    def is_temporal(self) -> bool:
        return self.strategy in TEMPORAL_STRATEGIES

    def validate(self):
        if self.strategy not in {"random_kfold", "stratified_kfold", "group_kfold", *TEMPORAL_STRATEGIES}:
            raise ValueError(f"Unknown validation strategy: {self.strategy}")
        if self.is_temporal and not self.timestamp_column:
            raise ValueError(f"Validation strategy '{self.strategy}' requires timestamp_column")
        if self.strategy == "group_kfold" and not self.group_columns:
            raise ValueError("Validation strategy 'group_kfold' requires group_columns")
        if self.strategy == "sliding_window" and not self.train_periods:
            raise ValueError("Validation strategy 'sliding_window' requires train_periods")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")
        if self.test_periods < 1:
            raise ValueError("test_periods must be >= 1")
        return self


# --------------------------------------------------------------------------- #
# Non-temporal strategies
# --------------------------------------------------------------------------- #


class RandomKFold:
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: int = 100):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    @property
    def strategy_name(self) -> str:
        return "random_kfold"

    def split(self, X, y=None, groups=None, timestamps=None):
        cv = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        yield from cv.split(_as_array(X))


class StratifiedKFold:
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: int = 100):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    @property
    def strategy_name(self) -> str:
        return "stratified_kfold"

    def split(self, X, y=None, groups=None, timestamps=None):
        if y is None:
            raise ValueError("stratified_kfold requires y")
        cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        yield from cv.split(_as_array(X), np.asarray(y))


class GroupKFold:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    @property
    def strategy_name(self) -> str:
        return "group_kfold"

    def split(self, X, y=None, groups=None, timestamps=None):
        if groups is None:
            raise ValueError("group_kfold requires groups")
        cv = _SkGroupKFold(n_splits=self.n_splits)
        yield from cv.split(_as_array(X), None if y is None else np.asarray(y), np.asarray(groups))


# --------------------------------------------------------------------------- #
# Temporal strategies
# --------------------------------------------------------------------------- #


class RollingOrigin:
    """Forward-chaining validation on a period axis.

    Periods are the sorted unique timestamps. For each cutoff the training set is
    every row whose period precedes the cutoff (expanding) or the last
    ``train_periods`` periods before it (sliding); the validation set is the
    ``test_periods`` periods that begin ``gap + horizon - 1`` periods after the
    cutoff. The cutoff then advances by ``step`` periods.

    ``TRAIN weeks 1..52 -> VALIDATE weeks 53..56``, ``TRAIN weeks 1..56 -> VALIDATE 57..60`` ...
    """

    def __init__(
        self,
        *,
        horizon: int = 1,
        gap: int = 0,
        min_train_periods: int | None = None,
        test_periods: int = 1,
        step: int | None = None,
        train_periods: int | None = None,
        max_splits: int | None = None,
        window: Literal["expanding", "sliding"] = "expanding",
    ):
        if window == "sliding" and not train_periods:
            raise ValueError("sliding window requires train_periods")
        self.horizon = horizon
        self.gap = gap
        self.min_train_periods = min_train_periods
        self.test_periods = test_periods
        self.step = step or test_periods
        self.train_periods = train_periods
        self.max_splits = max_splits
        self.window = window

    @property
    def strategy_name(self) -> str:
        return "rolling_origin" if self.window == "expanding" and self.train_periods is None else f"rolling_origin_{self.window}"

    def split(self, X, y=None, groups=None, timestamps=None):
        if timestamps is None:
            raise ValueError(f"{self.strategy_name} requires timestamps")
        period_of_row, periods = _period_index(timestamps)
        n_periods = len(periods)
        lead = self.gap + max(0, self.horizon - 1)
        min_train = self.min_train_periods or max(1, n_periods // 2)
        cutoffs = range(min_train, n_periods - lead - self.test_periods + 1, self.step)
        emitted = 0
        for cutoff in cutoffs:
            train_start = 0 if self.window == "expanding" else max(0, cutoff - int(self.train_periods))
            test_start = cutoff + lead
            test_end = test_start + self.test_periods
            train_mask = (period_of_row >= train_start) & (period_of_row < cutoff)
            test_mask = (period_of_row >= test_start) & (period_of_row < test_end)
            if not train_mask.any() or not test_mask.any():
                continue
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
            emitted += 1
            if self.max_splits and emitted >= self.max_splits:
                break
        if emitted == 0:
            raise ValueError(
                f"{self.strategy_name}: no valid windows (periods={n_periods}, min_train_periods={min_train}, "
                f"test_periods={self.test_periods}, lead={lead})"
            )


class ExpandingWindow(RollingOrigin):
    def __init__(self, **kwargs):
        kwargs.pop("window", None)
        kwargs.pop("train_periods", None)
        super().__init__(window="expanding", **kwargs)

    @property
    def strategy_name(self) -> str:
        return "expanding_window"


class SlidingWindow(RollingOrigin):
    def __init__(self, *, train_periods: int, **kwargs):
        kwargs.pop("window", None)
        super().__init__(window="sliding", train_periods=train_periods, **kwargs)

    @property
    def strategy_name(self) -> str:
        return "sliding_window"


# --------------------------------------------------------------------------- #
# Materialized folds shared by every experiment
# --------------------------------------------------------------------------- #


@dataclass
class FoldSet:
    """A frozen list of positional folds.

    It is sklearn-compatible (``split``/``get_n_splits``) so it can be passed as
    ``cv=`` to ``cross_val_score`` and ``RandomizedSearchCV``. Every experiment that
    reuses the same FoldSet is evaluated on identical rows.
    """

    folds: list[Fold]
    strategy_name: str
    n_rows: int
    index: pd.Index | None = None
    timestamps: pd.Series | None = None
    groups: pd.Series | None = None
    config: ValidationConfig | None = field(default=None, repr=False)

    @classmethod
    def from_strategy(cls, strategy: ValidationStrategy, X, y=None, groups=None, timestamps=None, config=None) -> FoldSet:
        n = len(X)
        folds = [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in strategy.split(X, y, groups, timestamps)]
        if not folds:
            raise ValueError("validation strategy produced no folds")
        ts = None if timestamps is None else pd.Series(np.asarray(timestamps)).reset_index(drop=True)
        gr = None if groups is None else pd.Series(np.asarray(groups)).reset_index(drop=True)
        idx = X.index if isinstance(X, pd.DataFrame | pd.Series) else None
        fs = cls(folds, strategy.strategy_name, n, idx, ts, gr, config)
        if ts is not None and strategy.strategy_name in TEMPORAL_STRATEGIES | {"rolling_origin_sliding", "rolling_origin_expanding"}:
            fs.assert_temporal_integrity()
        return fs

    def split(self, X=None, y=None, groups=None):
        if X is not None and len(X) != self.n_rows:
            raise ValueError(f"FoldSet was materialized for {self.n_rows} rows but received {len(X)}; experiments must share the same rows")
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    def __len__(self):
        return len(self.folds)

    @property
    def is_temporal(self) -> bool:
        return self.timestamps is not None and self.strategy_name.startswith(("rolling", "expanding", "sliding"))

    def assert_temporal_integrity(self):
        """Raise if any training row is on/after the earliest validation timestamp of its fold."""
        if self.timestamps is None:
            raise ValueError("no timestamps recorded")
        ts = self.timestamps.to_numpy()
        for i, (tr, te) in enumerate(self.folds):
            if len(tr) and len(te) and ts[tr].max() >= ts[te].min():
                raise ValueError(f"fold {i} trains on timestamp {ts[tr].max()} which is not before validation start {ts[te].min()}")
        return True

    def describe(self) -> pd.DataFrame:
        rows = []
        for i, (tr, te) in enumerate(self.folds):
            row = {"fold": i + 1, "train_rows": len(tr), "validation_rows": len(te)}
            if self.timestamps is not None:
                ts = self.timestamps
                row.update(
                    {
                        "train_start": ts.iloc[tr].min(),
                        "train_end": ts.iloc[tr].max(),
                        "validation_start": ts.iloc[te].min(),
                        "validation_end": ts.iloc[te].max(),
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Factories and helpers
# --------------------------------------------------------------------------- #


def build_strategy(config: ValidationConfig, task: str = "regression") -> ValidationStrategy:
    config.validate()
    s = config.strategy
    if s == "random_kfold":
        return RandomKFold(config.n_splits, config.n_repeats, config.random_state)
    if s == "stratified_kfold":
        return StratifiedKFold(config.n_splits, config.n_repeats, config.random_state)
    if s == "group_kfold":
        return GroupKFold(config.n_splits)
    common = {
        "horizon": config.horizon,
        "gap": config.gap,
        "min_train_periods": config.min_train_periods,
        "test_periods": config.test_periods,
        "step": config.step,
        "max_splits": config.max_splits,
    }
    if s == "rolling_origin":
        return RollingOrigin(**common, train_periods=config.train_periods, window="sliding" if config.train_periods else "expanding")
    if s == "expanding_window":
        return ExpandingWindow(**common)
    return SlidingWindow(train_periods=int(config.train_periods), **common)


def default_validation(task: str, n_splits: int = 5, n_repeats: int = 1, random_state: int = 100) -> ValidationConfig:
    return ValidationConfig(
        strategy="stratified_kfold" if task == "classification" else "random_kfold",
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def resolve_columns(frame: pd.DataFrame, config: ValidationConfig) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (groups, timestamps) series for ``frame`` according to ``config``."""
    timestamps = None
    groups = None
    if config.timestamp_column:
        if config.timestamp_column not in frame:
            raise KeyError(f"timestamp_column '{config.timestamp_column}' not in data")
        timestamps = frame[config.timestamp_column]
    if config.group_columns:
        missing = [c for c in config.group_columns if c not in frame]
        if missing:
            raise KeyError(f"group_columns missing from data: {missing}")
        groups = (
            frame[list(config.group_columns)].astype(str).agg("|".join, axis=1)
            if len(config.group_columns) > 1
            else frame[config.group_columns[0]]
        )
    return groups, timestamps


def materialize_folds(frame: pd.DataFrame, y, config: ValidationConfig, task: str) -> FoldSet:
    groups, timestamps = resolve_columns(frame, config)
    strategy = build_strategy(config, task)
    return FoldSet.from_strategy(strategy, frame, y, groups, timestamps, config)


def development_holdout_split(
    frame: pd.DataFrame, y, config: ValidationConfig, task: str, holdout_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split positional indices into (development, final holdout).

    Temporal strategies reserve the *last* periods; group strategies reserve whole
    groups; classification uses stratification; otherwise a seeded random split.
    """
    n = len(frame)
    frac = config.holdout_fraction if config.holdout_fraction is not None else holdout_fraction
    if frac <= 0 and not config.holdout_periods:
        return np.arange(n), np.array([], dtype=int)
    groups, timestamps = resolve_columns(frame, config)
    if config.is_temporal:
        period_of_row, periods = _period_index(timestamps)
        n_periods = len(periods)
        k = config.holdout_periods or max(1, math.ceil(n_periods * frac))
        if k >= n_periods:
            raise ValueError(f"holdout would consume all {n_periods} periods")
        cutoff = n_periods - k
        dev = np.flatnonzero(period_of_row < cutoff)
        hold = np.flatnonzero(period_of_row >= cutoff)
        return dev, hold
    if config.strategy == "group_kfold":
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=config.random_state)
        dev, hold = next(gss.split(np.zeros(n), None, np.asarray(groups)))
        return dev, hold
    strat = np.asarray(y) if task == "classification" and y is not None else None
    dev, hold = train_test_split(np.arange(n), test_size=frac, random_state=config.random_state, stratify=strat)
    return np.sort(dev), np.sort(hold)


def _as_array(X):
    return X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)


def _period_index(timestamps) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.Series(np.asarray(timestamps)).reset_index(drop=True)
    if ts.isna().any():
        raise ValueError("timestamps contain missing values")
    periods = np.sort(ts.unique())
    period_of_row = np.searchsorted(periods, ts.to_numpy())
    return period_of_row, periods
