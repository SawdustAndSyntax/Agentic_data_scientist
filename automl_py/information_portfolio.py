from __future__ import annotations

from itertools import combinations
import pandas as pd


def optimize_information_portfolio(
    voi: pd.DataFrame,
    annual_budget: float,
    max_candidates: int | None = None,
) -> dict:
    """
    Small exact optimizer for choosing a portfolio of candidate information
    investments under a first-year budget.

    This intentionally uses exhaustive search for transparency and avoids adding
    a solver dependency. Suitable for a shortlist of candidates (<= ~20).
    """
    if annual_budget < 0:
        raise ValueError("annual_budget must be >= 0")
    if voi.empty:
        return {"selected": [], "total_cost": 0.0, "total_net_value": 0.0}

    frame = voi.reset_index(drop=True)
    n = len(frame)
    if n > 20:
        # Greedy fallback by net-value-per-dollar for larger candidate lists.
        f = frame.copy()
        f["_ratio"] = f["first_year_net_value"] / f["first_year_cost"].replace(0, 1e-12)
        f = f.sort_values(["_ratio", "first_year_net_value"], ascending=False)
        selected, cost, net = [], 0.0, 0.0
        for _, row in f.iterrows():
            if row["first_year_net_value"] <= 0:
                continue
            if max_candidates is not None and len(selected) >= max_candidates:
                break
            if cost + row["first_year_cost"] <= annual_budget:
                selected.append(row["candidate"])
                cost += float(row["first_year_cost"])
                net += float(row["first_year_net_value"])
        return {"selected": selected, "total_cost": cost, "total_net_value": net}

    best = {"selected": [], "total_cost": 0.0, "total_net_value": 0.0}
    limit = n if max_candidates is None else min(n, max_candidates)

    for r in range(1, limit + 1):
        for idxs in combinations(range(n), r):
            subset = frame.iloc[list(idxs)]
            cost = float(subset["first_year_cost"].sum())
            net = float(subset["first_year_net_value"].sum())
            if cost <= annual_budget and net > best["total_net_value"]:
                best = {
                    "selected": subset["candidate"].tolist(),
                    "total_cost": cost,
                    "total_net_value": net,
                }
    return best
