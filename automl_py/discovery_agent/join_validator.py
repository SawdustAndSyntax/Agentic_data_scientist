from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .contracts import JoinPlan

ROW_EXPLOSION = "ROW_EXPLOSION"
LOW_COVERAGE = "LOW_COVERAGE"
DUPLICATE_KEYS = "DUPLICATE_KEYS"


@dataclass
class JoinValidation:
    left_rows: int
    right_rows: int
    output_rows: int
    matched_left_pct: float
    row_multiplier: float
    duplicate_key_risk: bool
    status: str  # "pass" | "invalid"
    warnings: list[str]
    reason_codes: list[str] = field(default_factory=list)
    null_join_rate: float = 0.0

    @property
    def valid(self) -> bool:
        return self.status == "pass"


class JoinValidator:
    """Row-explosion and coverage checks run *before* any model is trained.

    A join whose row multiplier exceeds ``max_row_multiplier`` or whose coverage
    falls below ``min_coverage`` is INVALID with an explicit reason code.
    """

    def __init__(self, max_row_multiplier: float = 1.20, min_coverage: float = 0.70):
        self.max_row_multiplier = max_row_multiplier
        self.min_coverage = min_coverage

    def validate_frames(self, left: pd.DataFrame, right: pd.DataFrame, left_keys: list[str], right_keys: list[str]) -> JoinValidation:
        warnings = []
        codes = []
        if len(left_keys) != len(right_keys) or not left_keys:
            raise ValueError("Join keys must be non-empty and have equal length")
        for c in left_keys:
            if c not in left:
                raise KeyError(f"Left join key missing: {c}")
        for c in right_keys:
            if c not in right:
                raise KeyError(f"Right join key missing: {c}")
        left_rows = left.reset_index(drop=False).rename(columns={"index": "__left_row_id"})
        merged = left_rows.merge(right, left_on=left_keys, right_on=right_keys, how="left", indicator=True, suffixes=("", "__candidate"))
        matched = merged.loc[merged["_merge"] == "both", "__left_row_id"].nunique()
        matched_pct = matched / max(1, len(left))
        multiplier = len(merged) / max(1, len(left))
        dup = bool(right.duplicated(right_keys, keep=False).any())
        null_rate = float(left[left_keys].isna().any(axis=1).mean()) if len(left) else 0.0
        if dup:
            warnings.append("candidate join keys are not unique")
            codes.append(DUPLICATE_KEYS)
        if multiplier > self.max_row_multiplier:
            warnings.append(f"row explosion detected ({multiplier:.2f}x)")
            codes.append(ROW_EXPLOSION)
        if matched_pct < self.min_coverage:
            warnings.append(f"low join coverage ({matched_pct:.1%})")
            codes.append(LOW_COVERAGE)
        status = "pass" if (ROW_EXPLOSION not in codes and LOW_COVERAGE not in codes) else "invalid"
        return JoinValidation(len(left), len(right), len(merged), matched_pct, multiplier, dup, status, warnings, codes, null_rate)

    def validate_plan(self, plan: JoinPlan) -> dict:
        explicit = all(r.source_keys and r.target_keys for r in plan.relationships)
        return {
            "path": " -> ".join(plan.path),
            "hops": len(plan.relationships),
            "confidence": plan.confidence,
            "explicit_keys": explicit,
            "warnings": list(plan.warnings),
            "status": "pass" if explicit and plan.confidence >= 0.7 else "review",
        }
