from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .contracts import JoinPlan

@dataclass
class JoinValidation:
    left_rows: int
    right_rows: int
    output_rows: int
    matched_left_pct: float
    row_multiplier: float
    duplicate_key_risk: bool
    status: str
    warnings: list[str]

class JoinValidator:
    def validate_frames(self, left: pd.DataFrame, right: pd.DataFrame, left_keys: list[str], right_keys: list[str]) -> JoinValidation:
        warnings=[]
        if len(left_keys)!=len(right_keys) or not left_keys:
            raise ValueError('Join keys must be non-empty and have equal length')
        for c in left_keys:
            if c not in left: raise KeyError(f'Left join key missing: {c}')
        for c in right_keys:
            if c not in right: raise KeyError(f'Right join key missing: {c}')
        l=left.reset_index(drop=False).rename(columns={'index':'__left_row_id'})
        merged=l.merge(right,left_on=left_keys,right_on=right_keys,how='left',indicator=True,suffixes=('','__candidate'))
        matched=merged.loc[merged['_merge']=='both','__left_row_id'].nunique()
        matched_pct=matched/max(1,len(left))
        multiplier=len(merged)/max(1,len(left))
        dup=bool(right.duplicated(right_keys,keep=False).any())
        if dup: warnings.append('candidate join keys are not unique')
        if multiplier>1.20: warnings.append(f'row explosion detected ({multiplier:.2f}x)')
        if matched_pct<0.70: warnings.append(f'low join coverage ({matched_pct:.1%})')
        status='pass' if multiplier<=1.20 and matched_pct>=0.70 else 'review'
        return JoinValidation(len(left),len(right),len(merged),matched_pct,multiplier,dup,status,warnings)

    def validate_plan(self, plan: JoinPlan) -> dict:
        explicit=all(r.source_keys and r.target_keys for r in plan.relationships)
        return {
            'path':' -> '.join(plan.path),
            'hops':len(plan.relationships),
            'confidence':plan.confidence,
            'explicit_keys':explicit,
            'warnings':list(plan.warnings),
            'status':'pass' if explicit and plan.confidence>=0.7 else 'review',
        }
