import numpy as np
import pandas as pd


class DataProfiler:
    def profile(self, df, target=None):
        rows = []
        n = len(df)
        for col in df.columns:
            s = df[col]
            nn = s.dropna()
            num = pd.api.types.is_numeric_dtype(s)
            rows.append(
                {
                    "feature": col,
                    "dtype": str(s.dtype),
                    "rows": n,
                    "missing_pct": float(s.isna().mean()),
                    "unique": int(s.nunique(dropna=True)),
                    "unique_pct": float(s.nunique(dropna=True) / max(len(nn), 1)),
                    "is_constant": bool(s.nunique(dropna=True) <= 1),
                    "is_high_cardinality": bool((not num) and s.nunique(dropna=True) > min(100, max(20, n * 0.2))),
                    "mean": float(s.mean()) if num and len(nn) else np.nan,
                    "std": float(s.std()) if num and len(nn) else np.nan,
                    "min": float(s.min()) if num and len(nn) else np.nan,
                    "max": float(s.max()) if num and len(nn) else np.nan,
                    "target": col == target,
                }
            )
        return pd.DataFrame(rows)
