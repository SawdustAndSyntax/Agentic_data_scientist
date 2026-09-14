"""The Experiment Judge.

A candidate is judged on paired, fold-by-fold differences against a baseline
evaluated on identical folds. An improvement greater than zero is not evidence:
the candidate must clear a practical-significance floor, an empirical noise
floor built from random control features, and a confidence interval that
excludes no effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

KEEP = "KEEP"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID = "INVALID"
REVIEW = "REVIEW"
DECISIONS = (KEEP, REJECT, INCONCLUSIVE, INVALID, REVIEW)


@dataclass(frozen=True)
class PairedUplift:
    n: int
    mean: float
    median: float
    std: float
    positive_share: float
    ci_low: float
    ci_high: float
    ci_level: float
    worst: float
    best: float
    uplifts: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uplifts"] = list(self.uplifts)
        return d


def paired_uplift(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    higher_is_better: bool,
    ci_level: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: int = 100,
) -> PairedUplift:
    b = np.asarray(baseline_scores, dtype=float)
    c = np.asarray(candidate_scores, dtype=float)
    if b.shape != c.shape:
        raise ValueError(f"baseline ({len(b)}) and candidate ({len(c)}) must be scored on identical folds")
    mask = np.isfinite(b) & np.isfinite(c)
    b, c = b[mask], c[mask]
    d = (c - b) if higher_is_better else (b - c)
    n = len(d)
    if n == 0:
        return PairedUplift(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ci_level, np.nan, np.nan, ())
    mean = float(d.mean())
    if n == 1:
        return PairedUplift(1, mean, mean, np.nan, float(d[0] > 0), np.nan, np.nan, ci_level, mean, mean, tuple(map(float, d)))
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = d[idx].mean(axis=1)
    alpha = (1 - ci_level) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return PairedUplift(
        n=n,
        mean=mean,
        median=float(np.median(d)),
        std=float(d.std(ddof=1)),
        positive_share=float((d > 0).mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=ci_level,
        worst=float(d.min()),
        best=float(d.max()),
        uplifts=tuple(map(float, d)),
    )


@dataclass
class ExperimentVerdict:
    decision: str
    uplift: PairedUplift | None
    baseline_mean: float
    candidate_mean: float
    required_gain: float
    minimum_absolute_gain: float
    minimum_relative_gain: float
    noise_threshold: float | None
    noise_quantile: float
    min_positive_share: float
    reasons: list[str] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    @property
    def established(self) -> bool:
        """True only when predictive uplift is established beyond noise and the practical floor."""
        return self.decision == KEEP

    @property
    def mean_uplift(self) -> float:
        return float("nan") if self.uplift is None else self.uplift.mean

    def to_dict(self) -> dict:
        d = {
            "decision": self.decision,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "required_gain": self.required_gain,
            "minimum_absolute_gain": self.minimum_absolute_gain,
            "minimum_relative_gain": self.minimum_relative_gain,
            "noise_threshold": self.noise_threshold,
            "noise_quantile": self.noise_quantile,
            "min_positive_share": self.min_positive_share,
            "reasons": list(self.reasons),
            "invalid_reasons": list(self.invalid_reasons),
            "review_reasons": list(self.review_reasons),
        }
        if self.uplift is not None:
            u = self.uplift.to_dict()
            d.update(
                {
                    "n_folds": u["n"],
                    "mean_uplift": u["mean"],
                    "median_uplift": u["median"],
                    "uplift_std": u["std"],
                    "positive_share": u["positive_share"],
                    "ci_low": u["ci_low"],
                    "ci_high": u["ci_high"],
                    "ci_level": u["ci_level"],
                    "worst_uplift": u["worst"],
                    "best_uplift": u["best"],
                    "fold_uplifts": u["uplifts"],
                }
            )
        return d

    def summary(self) -> str:
        if self.uplift is None:
            return f"{self.decision}: " + "; ".join(self.reasons or self.invalid_reasons)
        u = self.uplift
        return (
            f"{self.decision}: mean uplift {u.mean:.4f} [{u.ci_low:.4f}, {u.ci_high:.4f}] "
            f"positive share {u.positive_share:.0%}, required gain {self.required_gain:.4f}"
            + (f", noise P{int(self.noise_quantile * 100)} {self.noise_threshold:.4f}" if self.noise_threshold is not None else "")
        )


class ExperimentJudge:
    """Turn paired fold scores into a KEEP / REJECT / INCONCLUSIVE / INVALID / REVIEW decision.

    KEEP requires, on identical folds: mean uplift >= required gain, a bootstrap
    confidence interval whose lower bound is above zero, and a positive-fold
    share of at least ``min_positive_share``. ``required gain`` is the largest
    of the absolute floor, the relative floor and the noise-control threshold.
    REJECT means even the optimistic CI bound cannot clear the floor. Anything
    in between is INCONCLUSIVE. Validity failures are INVALID regardless of
    score, and a KEEP with outstanding review reasons becomes REVIEW.
    """

    def __init__(
        self,
        *,
        minimum_absolute_gain: float = 0.0,
        minimum_relative_gain: float = 0.0,
        noise_quantile: float = 0.95,
        ci_level: float = 0.95,
        min_positive_share: float = 0.6,
        min_folds: int = 2,
        n_bootstrap: int = 2000,
        random_state: int = 100,
    ):
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be in (0, 1)")
        self.minimum_absolute_gain = float(minimum_absolute_gain)
        self.minimum_relative_gain = float(minimum_relative_gain)
        self.noise_quantile = noise_quantile
        self.ci_level = ci_level
        self.min_positive_share = min_positive_share
        self.min_folds = min_folds
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def noise_threshold(self, noise_uplifts: Sequence[float] | None) -> float | None:
        if noise_uplifts is None:
            return None
        arr = np.asarray([u for u in noise_uplifts if np.isfinite(u)], dtype=float)
        if arr.size == 0:
            return None
        return float(max(0.0, np.quantile(arr, self.noise_quantile)))

    def evaluate(
        self,
        baseline_scores: Sequence[float],
        candidate_scores: Sequence[float],
        noise_scores: Sequence[float] | None = None,
        minimum_gain: float | None = None,
        *,
        higher_is_better: bool = True,
        invalid_reasons: Sequence[str] = (),
        review_reasons: Sequence[str] = (),
    ) -> ExperimentVerdict:
        """``noise_scores`` are the mean paired uplifts of random control features on the same folds."""
        min_abs = self.minimum_absolute_gain if minimum_gain is None else float(minimum_gain)
        b = np.asarray(baseline_scores, dtype=float)
        c = np.asarray(candidate_scores, dtype=float)
        baseline_mean = float(np.nanmean(b)) if b.size else float("nan")
        candidate_mean = float(np.nanmean(c)) if c.size else float("nan")
        noise_thr = self.noise_threshold(noise_scores)
        required = max(min_abs, self.minimum_relative_gain * abs(baseline_mean) if np.isfinite(baseline_mean) else 0.0, noise_thr or 0.0)
        verdict = ExperimentVerdict(
            decision=INCONCLUSIVE,
            uplift=None,
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            required_gain=float(required),
            minimum_absolute_gain=min_abs,
            minimum_relative_gain=self.minimum_relative_gain,
            noise_threshold=noise_thr,
            noise_quantile=self.noise_quantile,
            min_positive_share=self.min_positive_share,
            invalid_reasons=list(invalid_reasons),
            review_reasons=list(review_reasons),
        )
        if invalid_reasons:
            verdict.decision = INVALID
            verdict.reasons.append("experiment violates validity constraints: " + "; ".join(invalid_reasons))
            return verdict
        if b.shape != c.shape:
            verdict.decision = INVALID
            verdict.invalid_reasons.append(f"baseline and candidate scored on different folds ({len(b)} vs {len(c)})")
            verdict.reasons.append(verdict.invalid_reasons[-1])
            return verdict

        up = paired_uplift(
            b, c, higher_is_better=higher_is_better, ci_level=self.ci_level, n_bootstrap=self.n_bootstrap, random_state=self.random_state
        )
        verdict.uplift = up
        if up.n < self.min_folds:
            verdict.decision = INCONCLUSIVE
            verdict.reasons.append(f"only {up.n} paired fold(s); at least {self.min_folds} required to estimate uncertainty")
            return verdict

        clears_floor = up.mean >= required
        ci_excludes_zero = up.ci_low > 0
        stable = up.positive_share >= self.min_positive_share
        if clears_floor and ci_excludes_zero and stable:
            verdict.decision = REVIEW if review_reasons else KEEP
            verdict.reasons.append(
                f"mean uplift {up.mean:.4f} clears required gain {required:.4f}; CI [{up.ci_low:.4f}, {up.ci_high:.4f}] excludes zero"
            )
            if review_reasons:
                verdict.reasons.append("requires review: " + "; ".join(review_reasons))
            return verdict
        if up.ci_high <= required:
            verdict.decision = REJECT
            verdict.reasons.append(f"upper CI bound {up.ci_high:.4f} does not reach required gain {required:.4f}")
            return verdict
        verdict.decision = INCONCLUSIVE
        if not clears_floor:
            verdict.reasons.append(f"mean uplift {up.mean:.4f} below required gain {required:.4f} but CI reaches it")
        if not ci_excludes_zero:
            verdict.reasons.append(f"CI [{up.ci_low:.4f}, {up.ci_high:.4f}] includes zero")
        if not stable:
            verdict.reasons.append(f"positive-fold share {up.positive_share:.0%} below {self.min_positive_share:.0%}")
        return verdict
