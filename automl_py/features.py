import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression


class RatioFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ratios, epsilon=1e-9):
        self.ratios = ratios
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for name, (num, den) in self.ratios.items():
            d = X[den].astype(float)
            X[name] = X[num].astype(float) / d.where(d.abs() > self.epsilon, np.nan)
        return X


class RowStatistics(BaseEstimator, TransformerMixin):
    def __init__(self, columns, prefix="row", stats=("mean", "median", "std", "min", "max")):
        self.columns = list(columns)
        self.prefix = prefix
        self.stats = tuple(stats)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        b = X[self.columns].apply(pd.to_numeric, errors="coerce")
        f = {
            "mean": lambda d: d.mean(axis=1),
            "median": lambda d: d.median(axis=1),
            "std": lambda d: d.std(axis=1),
            "min": lambda d: d.min(axis=1),
            "max": lambda d: d.max(axis=1),
            "sum": lambda d: d.sum(axis=1),
        }
        for stat in self.stats:
            X[f"{self.prefix}_{stat}"] = f[stat](b)
        return X


class GroupStatisticsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, value_cols, stats=("mean", "median")):
        self.group_cols = list(group_cols)
        self.value_cols = list(value_cols)
        self.stats = tuple(stats)

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.global_ = {}
        self.maps_ = {}
        g = X.groupby(self.group_cols, dropna=False)
        for v in self.value_cols:
            self.global_[v] = {s: getattr(pd.to_numeric(X[v], errors="coerce"), s)() for s in self.stats}
            for s in self.stats:
                self.maps_[(v, s)] = g[v].agg(s)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for v in self.value_cols:
            for s in self.stats:
                name = "__".join([*self.group_cols, v, s])
                m = self.maps_[(v, s)]
                vals = (
                    X[self.group_cols[0]].map(m)
                    if len(self.group_cols) == 1
                    else pd.Series(m.reindex(pd.MultiIndex.from_frame(X[self.group_cols])).to_numpy(), index=X.index)
                )
                X[name] = vals.fillna(self.global_[v][s])
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, drop_original=False):
        self.columns = list(columns)
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns:
            s = pd.to_datetime(X[c], errors="coerce")
            X[f"{c}__year"] = s.dt.year
            X[f"{c}__quarter"] = s.dt.quarter
            X[f"{c}__month"] = s.dt.month
            X[f"{c}__week"] = s.dt.isocalendar().week.astype(float)
            X[f"{c}__dayofweek"] = s.dt.dayofweek
            if self.drop_original:
                X = X.drop(columns=[c])
        return X


def add_lag_features(df, value_cols, lags=(1, 2, 3), group_cols=None, sort_cols=None):
    out = df.copy()
    if sort_cols:
        out = out.sort_values(list(sort_cols)).copy()
    g = list(group_cols or [])
    for c in value_cols:
        for lag in lags:
            out[f"{c}__lag_{lag}"] = out.groupby(g, dropna=False)[c].shift(lag) if g else out[c].shift(lag)
    return out


def add_rolling_features(df, value_cols, windows=(3, 7, 14), stats=("mean", "median", "std"), group_cols=None, sort_cols=None, shift=1):
    out = df.copy()
    if sort_cols:
        out = out.sort_values(list(sort_cols)).copy()
    g = list(group_cols or [])
    for c in value_cols:
        shifted = out.groupby(g, dropna=False)[c].shift(shift) if g else out[c].shift(shift)
        for w in windows:
            if g:
                keys = [out[x] for x in g]
                roll = shifted.groupby(keys, dropna=False).rolling(w, min_periods=1)
            else:
                roll = shifted.rolling(w, min_periods=1)
            for stat in stats:
                vals = getattr(roll, stat)()
                if g:
                    vals = vals.reset_index(level=list(range(len(g))), drop=True)
                out[f"{c}__rolling_{w}_{stat}"] = vals
    return out


def make_pca(n_components=0.95, random_state=100):
    return PCA(n_components=n_components, svd_solver="full", random_state=random_state)


def make_feature_selector(task, k="all", method="f_test"):
    fn = (
        (f_classif if method == "f_test" else mutual_info_classif)
        if task == "classification"
        else (f_regression if method == "f_test" else mutual_info_regression)
    )
    return SelectKBest(fn, k=k)
