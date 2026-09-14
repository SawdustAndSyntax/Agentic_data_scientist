import re
import numpy as np
import pandas as pd

PATTERNS = [
    r"\bactual\b",
    r"\bfinal\b",
    r"\bclosed\b",
    r"\bresolved\b",
    r"\boutcome\b",
    r"\bresult\b",
    r"\bpost\b",
    r"\bdelivered\b",
    r"\bcompletion\b",
    r"\bcompleted\b",
]


class LeakageDetector:
    def detect(self, df, target, corr_threshold=0.98):
        y = df[target]
        rows = []
        for col in df.columns:
            if col == target:
                continue
            reasons = []
            score = 0
            name = col.lower().replace("_", " ")
            if any(re.search(p, name) for p in PATTERNS):
                reasons.append("name suggests post-outcome or target-derived information")
                score += 2
            x = df[col]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                pair = pd.concat([x, y], axis=1).dropna()
                if len(pair) >= 5:
                    corr = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                    if np.isfinite(corr) and corr >= corr_threshold:
                        reasons.append(f"near-perfect numeric relationship with target (|r|={corr:.3f})")
                        score += 3
            rows.append(
                {
                    "feature": col,
                    "risk_score": score,
                    "risk": "high" if score >= 3 else "medium" if score >= 2 else "low",
                    "reasons": "; ".join(reasons),
                }
            )
        return pd.DataFrame(rows).sort_values(["risk_score", "feature"], ascending=[False, True])
