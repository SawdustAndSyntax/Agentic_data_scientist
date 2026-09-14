import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class MissingnessAnalyzer:
    def analyze(self, df, target=None, random_state=100):
        rows = []
        for col in df.columns:
            if col == target or not df[col].isna().any():
                continue
            y = df[col].isna().astype(int)
            X = df.drop(columns=[col] + ([target] if target and target in df else []))
            auc = np.nan
            if y.nunique() == 2 and min(y.value_counts()) >= 5 and X.shape[1] > 0:
                nums = X.select_dtypes(include=["number", "bool"]).columns.tolist()
                cats = [c for c in X.columns if c not in nums]
                prep = ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), nums),
                        (
                            "cat",
                            Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore"))]),
                            cats,
                        ),
                    ]
                )
                pipe = Pipeline(
                    [
                        ("prep", prep),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=120, max_depth=6, class_weight="balanced", random_state=random_state, n_jobs=1
                            ),
                        ),
                    ]
                )
                folds = min(5, int(min(y.value_counts())))
                try:
                    auc = float(
                        np.mean(
                            cross_val_score(
                                pipe, X, y, scoring="roc_auc", cv=StratifiedKFold(folds, shuffle=True, random_state=random_state), n_jobs=1
                            )
                        )
                    )
                except Exception:
                    pass
            interp = (
                "insufficient data"
                if not np.isfinite(auc)
                else (
                    "strongly systematic"
                    if auc >= 0.8
                    else "moderately systematic"
                    if auc >= 0.65
                    else "weakly predictable / closer to random"
                )
            )
            rows.append(
                {
                    "feature": col,
                    "missing_pct": float(y.mean()),
                    "missingness_predictability_auc": auc,
                    "interpretation": interp,
                    "recommend_missing_indicator": bool(y.mean() >= 0.01 and (not np.isfinite(auc) or auc >= 0.6)),
                }
            )
        return pd.DataFrame(rows)
