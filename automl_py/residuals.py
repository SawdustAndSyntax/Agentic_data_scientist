import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def regression_residual_diagnostics(X, y_true, y_pred, top_n=15):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    rows = []
    for col in X.select_dtypes(include=["number", "bool"]).columns:
        s = X[col]
        mask = s.notna() & np.isfinite(residual)
        if mask.sum() < 10 or s[mask].nunique() <= 1:
            continue
        corr = np.corrcoef(s[mask].astype(float), residual[mask])[0, 1]
        try:
            mi = mutual_info_regression(s[mask].to_numpy().reshape(-1, 1), residual[mask], random_state=100)[0]
        except (ValueError, TypeError) as exc:
            warnings.warn(f"mutual information failed for {col}: {exc}", RuntimeWarning, stacklevel=2)
            mi = np.nan
        rows.append(
            {
                "feature": col,
                "residual_correlation": float(corr) if np.isfinite(corr) else np.nan,
                "abs_residual_correlation": abs(float(corr)) if np.isfinite(corr) else np.nan,
                "residual_mutual_information": float(mi) if np.isfinite(mi) else np.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["residual_mutual_information", "abs_residual_correlation"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["feature", "residual_correlation", "abs_residual_correlation", "residual_mutual_information"])
    )
