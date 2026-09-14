from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .agent import DataDiscoveryAgent
from .contracts import DiscoveryRequest
from .join_validator import JoinValidator


@dataclass
class DiscoveryExperimentResult:
    discovery_report: pd.DataFrame
    tested_candidates: pd.DataFrame


class PredictiveDiscoveryLoop:
    """Bridge discovery candidates into controlled modeling experiments.

    `candidate_loader` and `experiment_runner` are dependency-injected so warehouse
    extraction and the predictive scientist remain independently governable/testable.
    """

    def __init__(self, discovery_agent: DataDiscoveryAgent):
        self.agent = discovery_agent

    def run(
        self,
        request: DiscoveryRequest,
        *,
        anchor_entities: list[str],
        base_df: pd.DataFrame,
        candidate_loader,
        experiment_runner,
        top_n: int = 3,
    ) -> DiscoveryExperimentResult:
        candidates = self.agent.discover(request, anchor_entities=anchor_entities, profile=True)
        report = self.agent.to_frame(candidates)
        rows = []
        baseline = experiment_runner(base_df)
        base_score = float(baseline["score"])
        for c in candidates[:top_n]:
            try:
                extra, left_keys, right_keys = candidate_loader(c)
                jv = JoinValidator().validate_frames(base_df, extra, list(left_keys), list(right_keys))
                if jv.status != "pass":
                    rows.append(
                        {
                            "candidate": c.entity.qualified_name,
                            "status": "rejected_join",
                            "baseline_score": base_score,
                            "candidate_score": None,
                            "uplift": None,
                            "join_coverage": jv.matched_left_pct,
                            "row_multiplier": jv.row_multiplier,
                            "reason": "; ".join(jv.warnings),
                        }
                    )
                    continue
                merged = base_df.merge(extra, left_on=list(left_keys), right_on=list(right_keys), how="left", suffixes=("", "__candidate"))
                outcome = experiment_runner(merged)
                score = float(outcome["score"])
                higher = bool(outcome.get("higher_is_better", True))
                uplift = (score - base_score) if higher else (base_score - score)
                rows.append(
                    {
                        "candidate": c.entity.qualified_name,
                        "status": "keep" if uplift > 0 else "reject_no_uplift",
                        "baseline_score": base_score,
                        "candidate_score": score,
                        "uplift": uplift,
                        "join_coverage": jv.matched_left_pct,
                        "row_multiplier": jv.row_multiplier,
                        "reason": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "candidate": c.entity.qualified_name,
                        "status": "error",
                        "baseline_score": base_score,
                        "candidate_score": None,
                        "uplift": None,
                        "join_coverage": None,
                        "row_multiplier": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return DiscoveryExperimentResult(report, pd.DataFrame(rows))
