"""Tests designed to fail if scientific integrity breaks.

Each test maps to a requirement in the corrective sprint: temporal validation,
feature availability, the locked holdout, noise controls, paired uplift with
confidence intervals, join validity, experiment memory, the autonomous loop and
Value-of-Information integrity.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from automl_py import (
    INCONCLUSIVE,
    INVALID,
    KEEP,
    REJECT,
    REVIEW,
    UNKNOWN,
    AutoML,
    AutoMLConfig,
    AutonomousAutoMLScientist,
    BusinessValueModel,
    CandidateFeatureSet,
    DataCost,
    DataDiscoveryAgent,
    Diagnosis,
    DiscoveryRequest,
    ExperimentJudge,
    ExperimentMemory,
    FeatureAvailabilityRegistry,
    FeatureMetadata,
    FoldSet,
    HoldoutAlreadyEvaluated,
    HoldoutLocked,
    HypothesisGenerator,
    InMemoryCandidateSource,
    InMemoryCatalogAdapter,
    JoinValidator,
    PredictiveDiscoveryLoop,
    PredictiveDiscoveryOrchestrator,
    SemanticEntity,
    SemanticField,
    SemanticRelationship,
    SplitConformalRegressor,
    StopConfig,
    UpliftNotEstablished,
    ValidationConfig,
    ValueOfInformationEngine,
    fold_scores,
    materialize_folds,
    model_stability,
    optimize_information_portfolio,
    run_paired_experiment,
)
from automl_py.orchestrator import MAX_ITERATIONS, NO_CANDIDATES, NO_MEANINGFUL_IMPROVEMENT

warnings.filterwarnings("ignore")


def fast_config(tmp_path, **kw):
    base = {
        "task": "regression",
        "metric": "rmse",
        "preprocessors": ("original",),
        "imputation_strategies": ("median",),
        "feature_selection": ("none",),
        "models": ("ridge",),
        "tune": False,
        "cv_folds": 4,
        "n_jobs": 1,
        "save_models": False,
        "run_adversarial_validation": False,
        "stability_repeats": 3,
        "noise_controls": 4,
        "artifact_dir": str(tmp_path),
    }
    base.update(kw)
    return AutoMLConfig(**base)


def ridge_factory(X):
    return Pipeline([("m", Ridge())])


def panel(n_weeks=100, n_stores=3, seed=0):
    rng = np.random.default_rng(seed)
    weeks = np.repeat(np.arange(1, n_weeks + 1), n_stores)
    stores = np.tile(np.arange(1, n_stores + 1), n_weeks)
    n = len(weeks)
    x = rng.normal(size=n)
    temp = 20 + 10 * np.sin(2 * np.pi * weeks / 52) + rng.normal(0, 2, n)
    promo = rng.binomial(1, 0.3, n)
    y = 50 + 3 * x + 2.5 * temp + 15 * promo + rng.normal(0, 2, n)
    df = pd.DataFrame({"week": weeks, "store_id": stores, "x": x, "sales": y})
    extras = {"temp": temp, "promo": promo, "noise": rng.normal(size=n)}
    return df, extras


# --------------------------------------------------------------------------------------------- #
# 1. Validation strategy
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cfg",
    [
        ValidationConfig(
            strategy="rolling_origin", timestamp_column="week", group_columns=["store_id"], min_train_periods=30, test_periods=10, step=10
        ),
        ValidationConfig(
            strategy="expanding_window", timestamp_column="week", min_train_periods=30, test_periods=5, step=15, horizon=3, gap=1
        ),
        ValidationConfig(
            strategy="sliding_window", timestamp_column="week", train_periods=25, min_train_periods=25, test_periods=5, step=20
        ),
    ],
)
def test_temporal_strategies_never_train_on_or_after_validation(cfg):
    df, _ = panel()
    folds = materialize_folds(df, df.sales, cfg, "regression")
    assert len(folds) >= 2
    assert folds.assert_temporal_integrity()
    ts = df.week.to_numpy()
    lead = cfg.gap + max(0, cfg.horizon - 1)
    for tr, te in folds:
        assert ts[tr].max() < ts[te].min()
        assert ts[te].min() - ts[tr].max() >= lead + 1
        if cfg.strategy == "sliding_window":
            assert len(np.unique(ts[tr])) <= cfg.train_periods
    # every fold set is materialized once and refuses a frame of a different size
    with pytest.raises(ValueError):
        list(folds.split(df.iloc[:10]))


def test_foldset_detects_future_training_rows():
    ts = pd.Series(np.arange(10))
    bad = FoldSet([(np.array([0, 1, 2, 7]), np.array([3, 4]))], "rolling_origin", 10, timestamps=ts)
    with pytest.raises(ValueError, match="not before validation start"):
        bad.assert_temporal_integrity()


def test_group_kfold_keeps_groups_apart():
    df, _ = panel()
    cfg = ValidationConfig(strategy="group_kfold", group_columns=["store_id"], n_splits=3)
    folds = materialize_folds(df, df.sales, cfg, "regression")
    for tr, te in folds:
        assert not set(df.store_id.iloc[tr]) & set(df.store_id.iloc[te])


def test_random_cv_is_flattered_by_temporal_leakage_and_rolling_origin_is_not():
    """A random walk with a time index: interpolation looks great under shuffled CV and fails forward."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"t": np.arange(n), "y": np.cumsum(rng.normal(size=n))})
    tree = Pipeline([("m", DecisionTreeRegressor(max_depth=8, random_state=0))])
    rand = materialize_folds(df, df.y, ValidationConfig(strategy="random_kfold", n_splits=5), "regression")
    roll = materialize_folds(
        df,
        df.y,
        ValidationConfig(strategy="rolling_origin", timestamp_column="t", min_train_periods=200, test_periods=40, step=40),
        "regression",
    )
    r2_random = np.mean(fold_scores(tree, df[["t"]], df.y, rand, "r2"))
    r2_rolling = np.mean(fold_scores(tree, df[["t"]], df.y, roll, "r2"))
    assert r2_random > 0.8
    assert r2_rolling < 0.5
    assert r2_random - r2_rolling > 0.5


def test_validation_strategy_propagates_to_every_experiment(tmp_path):
    df, _extras = panel()
    cfg = fast_config(
        tmp_path,
        validation=ValidationConfig(
            strategy="rolling_origin", timestamp_column="week", group_columns=["store_id"], min_train_periods=40, test_periods=10, step=10
        ),
    )
    r = AutoML(cfg).fit(df, "sales")
    folds = r.folds["sales"]
    assert folds.strategy_name == "rolling_origin"
    assert (r.results.validation_strategy == "rolling_origin").all()
    assert r.results.n_folds.iloc[0] == len(folds)
    assert "week" not in r.feature_columns["sales"]
    # holdout is the last periods only
    hold_weeks = df.loc[r.holdouts["sales"].index, "week"]
    dev_weeks = df.loc[r.development_index["sales"], "week"]
    assert hold_weeks.min() > dev_weeks.max()
    # stability and family experiments reuse the identical folds
    X, y = r.development_frame("sales", df)
    stab = model_stability(r.best_model("sales"), X, y, task="regression", metric="rmse", folds=folds)
    assert stab["score_summary"]["source"] == "rolling_origin"
    assert stab["score_summary"]["n_partitions"] == len(folds)


# --------------------------------------------------------------------------------------------- #
# 2. Feature availability is distinct from validation and UNKNOWN is not safe
# --------------------------------------------------------------------------------------------- #
def test_feature_availability_states():
    reg = FeatureAvailabilityRegistry()
    reg.register_metadata(
        FeatureMetadata("temperature_forecast", event_time="2026-07-01", available_time="2026-06-30", source="weather_provider")
    )
    reg.register_metadata(FeatureMetadata("final_invoice", event_time="2026-07-01", available_time="2026-07-15"))
    audit = reg.audit(["temperature_forecast", "final_invoice", "mystery"], prediction_time="2026-07-01").set_index("feature")
    assert audit.loc["temperature_forecast", "availability"] == "available"
    assert audit.loc["final_invoice", "availability"] == "unavailable" and audit.loc["final_invoice", "temporal_leakage_risk"]
    assert audit.loc["mystery", "availability"] == UNKNOWN and audit.loc["mystery", "unknown_availability"]
    assert audit.loc["mystery", "available_at_prediction"] is None or pd.isna(audit.loc["mystery", "available_at_prediction"])
    df = pd.DataFrame({"temperature_forecast": [1.0], "final_invoice": [2.0], "mystery": [3.0]})
    with pytest.raises(ValueError):
        reg.enforce(df, strict=True)
    clean, _ = reg.enforce(df, strict=False)
    assert "final_invoice" not in clean and "mystery" in clean  # unknown is kept for review, never silently dropped or trusted
    with pytest.raises(ValueError, match="unknown"):
        reg.enforce(df, strict=False, unknown_policy="strict")
    judge = ExperimentJudge()
    strong = ([10, 10, 10, 10], [5, 5, 5, 5.5])
    assert judge.evaluate(*strong, higher_is_better=False, invalid_reasons=["TEMPORAL_LEAKAGE"]).decision == INVALID
    assert judge.evaluate(*strong, higher_is_better=False, review_reasons=["availability unknown"]).decision == REVIEW
    assert judge.evaluate(*strong, higher_is_better=False).decision == KEEP


def test_discovery_candidate_temporal_validity_is_not_hardcoded():
    e_known = SemanticEntity("w", "ext.weather", "memory", description="weather temperature", metadata={"availability_offset": "-1d"})
    e_late = SemanticEntity(
        "inv", "erp.invoice", "memory", description="weather invoice temperature", metadata={"availability_offset": "+14d"}
    )
    e_unknown = SemanticEntity("u", "ext.unknown", "memory", description="weather temperature humidity")
    agent = DataDiscoveryAgent(InMemoryCatalogAdapter([e_known, e_late, e_unknown]))
    out = {c.entity.qualified_name: c for c in agent.discover(DiscoveryRequest("sales", "weather temperature"), profile=False)}
    assert out["ext.weather"].temporal_validity == 1.0 and out["ext.weather"].availability == "available"
    assert out["erp.invoice"].temporal_validity == 0.0 and out["erp.invoice"].availability == "unavailable"
    assert out["ext.unknown"].temporal_validity is None and out["ext.unknown"].availability == UNKNOWN
    assert out["ext.weather"].score > out["ext.unknown"].score > out["erp.invoice"].score


# --------------------------------------------------------------------------------------------- #
# 3. Locked holdout
# --------------------------------------------------------------------------------------------- #
def test_final_holdout_is_locked_and_single_use(tmp_path):
    df, _ = panel()
    cfg = fast_config(tmp_path, models=("ridge", "tree"))
    r = AutoML(cfg).fit(df, "sales")
    hold = r.holdouts["sales"]
    # only the champion was evaluated; candidates carry development scores only
    assert "holdout_performance" not in r.results.columns
    assert len(hold.evaluations) == 1 and hold.evaluations[0].model_tag == r.best_row("sales").tag
    with pytest.raises(HoldoutAlreadyEvaluated):
        hold.evaluate(r.models[r.results.tag.iloc[1]], "challenger")
    log = hold.audit()
    assert (log.outcome == "refused_repeat").sum() == 1
    assert len(hold.evaluations) == 1
    # out-of-fold rows used for diagnostics never overlap the holdout
    oof_index = r.predictions[r.best_row("sales").tag].index
    assert not set(oof_index) & set(hold.index)


def test_holdout_cannot_be_evaluated_before_champion_lock(tmp_path):
    df, _ = panel()
    r = AutoML(fast_config(tmp_path, evaluate_holdout=False)).fit(df, "sales")
    hold = r.holdouts["sales"]
    hold.champion_locked = False
    with pytest.raises(HoldoutLocked):
        hold.evaluate(r.best_model("sales"))
    assert not hold.evaluated


# --------------------------------------------------------------------------------------------- #
# 4-7. Judge: noise controls, strong signal, inconclusive, confidence intervals
# --------------------------------------------------------------------------------------------- #
def signal_problem(seed=3, n=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 3 * x + 2 * z + rng.normal(0, 1.0, n)
    X = pd.DataFrame({"x": x, "z": z, "lucky": rng.normal(size=n), "weak": z * 0.05 + rng.normal(0, 1.0, n)})
    return X, pd.Series(y)


def test_random_feature_is_not_kept():
    X, y = signal_problem()
    folds = materialize_folds(X, y, ValidationConfig(strategy="random_kfold", n_splits=5), "regression")
    res = run_paired_experiment("lucky", ridge_factory, X, y, folds, "rmse", ["x"], ["lucky"], judge=ExperimentJudge(), n_noise_controls=6)
    assert res.decision in {REJECT, INCONCLUSIVE}
    assert res.decision != KEEP
    assert res.verdict.noise_threshold is not None


def test_strong_signal_is_kept_with_stable_paired_uplift():
    X, y = signal_problem()
    folds = materialize_folds(X, y, ValidationConfig(strategy="random_kfold", n_splits=5), "regression")
    judge = ExperimentJudge(minimum_absolute_gain=0.05)
    res = run_paired_experiment("z", ridge_factory, X, y, folds, "rmse", ["x"], ["z"], judge=judge, n_noise_controls=6)
    u = res.verdict.uplift
    assert res.decision == KEEP
    assert u.mean > 0 and u.ci_low > 0 and u.positive_share == 1.0
    assert u.mean > res.verdict.noise_threshold and u.mean > judge.minimum_absolute_gain
    assert u.n == 5 and len(u.uplifts) == 5
    assert res.verdict.required_gain == max(judge.minimum_absolute_gain, res.verdict.noise_threshold)


def test_minimum_gain_floor_is_enforced():
    judge = ExperimentJudge(minimum_absolute_gain=1.0)
    base = [100.0, 100.0, 100.0, 100.0, 100.0]
    tiny = [99.999, 99.998, 99.999, 99.999, 99.998]
    v = judge.evaluate(base, tiny, higher_is_better=False)
    assert v.decision == REJECT
    relative = ExperimentJudge(minimum_relative_gain=0.01).evaluate(base, [99.5] * 5, higher_is_better=False)
    assert relative.required_gain == pytest.approx(1.0) and relative.decision == REJECT


def test_inconclusive_when_confidence_interval_crosses_no_effect():
    judge = ExperimentJudge(minimum_absolute_gain=2.0)
    base = [114.0, 110.0, 118.0, 109.0, 115.0]
    cand = [110.0, 113.0, 113.0, 111.0, 113.0]  # paired diffs +4, -3, +5, -2, +2 -> mean 1.2, CI crosses zero
    v = judge.evaluate(base, cand, higher_is_better=False)
    assert v.uplift.mean == pytest.approx(1.2)
    assert v.uplift.ci_low < 0 < v.uplift.ci_high
    assert v.decision == INCONCLUSIVE
    strong = judge.evaluate(base, [101.0, 103.0, 104.0, 103.0, 101.0], noise_scores=[0.4, 1.3, 1.8, 0.2, 0.9], higher_is_better=False)
    assert strong.decision == KEEP and strong.uplift.ci_low > 0
    assert strong.noise_threshold == pytest.approx(np.quantile([0.4, 1.3, 1.8, 0.2, 0.9], 0.95))
    d = strong.to_dict()
    for key in (
        "mean_uplift",
        "median_uplift",
        "uplift_std",
        "positive_share",
        "ci_low",
        "ci_high",
        "worst_uplift",
        "best_uplift",
        "p_value",
        "ci_method",
    ):
        assert key in d


def test_corrected_interval_is_wider_than_naive_bootstrap_on_dependent_folds():
    """Nadeau-Bengio inflates the variance for overlapping training sets; the spec's
    worked example with one weak fold (+1) is honest-INCONCLUSIVE, not KEEP."""
    base = [114.0, 110.0, 118.0, 109.0, 115.0]
    cand = [101.0, 103.0, 104.0, 108.0, 101.0]  # diffs +13, +7, +14, +1, +14
    corrected = ExperimentJudge(minimum_absolute_gain=2.0).evaluate(base, cand, higher_is_better=False)
    naive = ExperimentJudge(minimum_absolute_gain=2.0, ci_method="bootstrap").evaluate(base, cand, higher_is_better=False)
    u = corrected.uplift
    assert u.ci_method == "nadeau_bengio" and (u.ci_high - u.ci_low) > (u.bootstrap_ci_high - u.bootstrap_ci_low)
    assert u.p_value < 0.05 and u.ci_low < 0  # one-sided p is small but the two-sided interval touches zero
    assert corrected.decision == INCONCLUSIVE and naive.decision == KEEP
    # a candidate with no variance in its uplift is decided without a degenerate interval
    flat = ExperimentJudge().evaluate([10.0, 10.0, 10.0], [8.0, 8.0, 8.0], higher_is_better=False)
    assert flat.decision == KEEP and flat.uplift.ci_low == pytest.approx(2.0)


def test_benjamini_hochberg_downgrades_lucky_candidates():
    from automl_py.judge import benjamini_hochberg

    assert benjamini_hochberg([0.001, 0.02, 0.04, 0.5], q=0.05) == [True, True, False, False]
    judge = ExperimentJudge()
    base = [10.0, 10.0, 10.0, 10.0, 10.0]
    strong = judge.evaluate(base, [7.0, 7.2, 6.9, 7.1, 7.0], higher_is_better=False)
    weak = judge.evaluate(base, [9.0, 9.6, 8.9, 9.7, 9.3], higher_is_better=False)
    assert strong.decision == KEEP and weak.decision == KEEP
    many = [judge.evaluate(base, [9.0, 9.6, 8.9, 9.7, 9.3], higher_is_better=False) for _ in range(30)]
    survivors = judge.control_false_discoveries([strong, weak, *many], q=0.001)
    assert survivors[0] is True and strong.decision == KEEP
    assert weak.decision == INCONCLUSIVE and "Benjamini-Hochberg" in weak.reasons[-1]


def test_judge_requires_identical_folds():
    v = ExperimentJudge().evaluate([1, 2, 3], [1, 2], higher_is_better=True)
    assert v.decision == INVALID


# --------------------------------------------------------------------------------------------- #
# Join validity and temporal joins in the discovery loop
# --------------------------------------------------------------------------------------------- #
def catalog():
    sales = SemanticEntity("sales", "a.sales", "memory", "table", "weekly store sales", [SemanticField("store_id", role="key")])
    weather = SemanticEntity(
        "weather", "ext.weather", "memory", "table", "weather temperature humidity", [SemanticField("store_id", role="key")]
    )
    rel = SemanticRelationship("s_w", "a.sales", "ext.weather", ("store_id",), ("store_id",))
    return DataDiscoveryAgent(InMemoryCatalogAdapter([sales, weather], [rel]))


def test_row_explosion_invalidates_before_training():
    left = pd.DataFrame({"store_id": [1, 2, 3], "sales": [10, 20, 30]})
    exploded = pd.DataFrame({"store_id": [1] * 20 + [2, 3], "temperature": range(22)})
    v = JoinValidator().validate_frames(left, exploded, ["store_id"], ["store_id"])
    assert v.status == "invalid" and "ROW_EXPLOSION" in v.reason_codes
    calls = []

    def runner(df):
        calls.append(len(df))
        return {"scores": [0.6, 0.62, 0.58], "higher_is_better": True}

    loop = PredictiveDiscoveryLoop(catalog())
    out = loop.run(
        DiscoveryRequest("sales", "weather temperature"),
        anchor_entities=["a.sales"],
        base_df=left,
        candidate_loader=lambda c: (exploded, ["store_id"], ["store_id"]),
        experiment_runner=runner,
        top_n=1,
    )
    row = out.tested_candidates.iloc[0]
    assert row.decision == INVALID and row.reason_code == "ROW_EXPLOSION"
    assert calls == [3]  # baseline only; the exploded candidate never reached the runner


def test_post_outcome_candidate_rejected_even_when_it_looks_perfect():
    left = pd.DataFrame({"store_id": [1, 2, 3], "sales": [10, 20, 30]})
    leak = pd.DataFrame({"store_id": [1, 2, 3], "final_sales": [10, 20, 30]})
    registry = FeatureAvailabilityRegistry({"final_sales": "+7d"})
    memory = ExperimentMemory()
    loop = PredictiveDiscoveryLoop(catalog(), feature_availability=registry, memory=memory)
    out = loop.run(
        DiscoveryRequest("sales", "weather temperature"),
        anchor_entities=["a.sales"],
        base_df=left,
        candidate_loader=lambda c: (leak, ["store_id"], ["store_id"]),
        experiment_runner=lambda df: {"scores": [1.0, 1.0, 1.0] if "final_sales" in df else [0.5, 0.5, 0.5], "higher_is_better": True},
        top_n=1,
    )
    row = out.tested_candidates.iloc[0]
    assert row.decision == INVALID and row.reason_code == "TEMPORAL_LEAKAGE"
    assert memory.records[0].decision == INVALID


# --------------------------------------------------------------------------------------------- #
# Experiment memory
# --------------------------------------------------------------------------------------------- #
def test_memory_blocks_identical_failed_experiment_without_new_evidence():
    gen = HypothesisGenerator()
    diag = Diagnosis(
        target="sales", metric="rmse", baseline_scores=[10, 11], current_columns=["x", "price"], context="weekly store sales demand"
    )
    memory = ExperimentMemory()
    first = gen.generate(diag, memory)
    weather = next(h for h in first if h.concept == "weather")
    memory.add(
        iteration=1,
        hypothesis=weather,
        candidate="weather_daily",
        candidate_columns=["temperature"],
        decision=REJECT,
        validation_strategy="rolling_origin",
        base_columns=["x", "price"],
    )
    again = gen.generate(diag, memory)
    assert all(h.concept != "weather" for h in again)
    # new evidence (a seasonal residual pattern) re-enables the hypothesis
    rf = pd.DataFrame({"residual": np.r_[np.full(50, 1.0), np.full(50, 5.0)]})
    diag2 = Diagnosis(**{**diag.__dict__, "residual_frame": rf, "timestamps": pd.Series(np.arange(100))})
    assert any(h.concept == "weather" for h in gen.generate(diag2, memory))
    # candidate-level memory: identical fingerprint + evidence is skipped, INVALID is sticky regardless of baseline
    fp = ExperimentMemory.fingerprint(["temperature"], "rolling_origin", ["x", "price"])
    cand_fp = ExperimentMemory.candidate_fingerprint(["temperature"])
    assert memory.should_skip(fp, weather.evidence_hash)[0]
    assert memory.should_skip(fp, "different-evidence")[0]  # same candidate, folds and baseline: outcome is deterministic
    new_base = ExperimentMemory.fingerprint(["temperature"], "rolling_origin", ["x", "price", "promo"])
    assert memory.should_skip(new_base, weather.evidence_hash, cand_fp)[0]  # new baseline but identical evidence
    assert not memory.should_skip(new_base, "different-evidence", cand_fp)[0]  # new baseline and new evidence -> re-test allowed
    memory.add(
        iteration=1,
        hypothesis=weather,
        candidate="leak",
        candidate_columns=["final"],
        decision=INVALID,
        validation_strategy="rolling_origin",
    )
    assert memory.should_skip("other-baseline", None, ExperimentMemory.candidate_fingerprint(["final"]))[0]
    assert len(memory.by_decision(REJECT)) == 1 and len(memory.by_decision(INVALID)) == 1  # negative results preserved


def test_unknown_signal_hypothesis_when_nothing_defensible():
    gen = HypothesisGenerator(library={})
    diag = Diagnosis(target="y", metric="rmse", baseline_scores=[1.0], current_columns=["a"])
    hyps = gen.generate(diag)
    assert len(hyps) == 1 and hyps[0].kind == "unknown" and hyps[0].concept == "UNKNOWN_SIGNAL"


# --------------------------------------------------------------------------------------------- #
# Autonomous loop
# --------------------------------------------------------------------------------------------- #
def test_autonomous_loop_discovers_keeps_and_stops(tmp_path):
    df, extras = panel(n_weeks=120, n_stores=4, seed=7)
    idx = df.index
    source = InMemoryCandidateSource(
        [
            CandidateFeatureSet(
                "weather_daily",
                pd.DataFrame({"temperature": extras["temp"]}, index=idx),
                ("weather", "temperature"),
                availability={"temperature": "available"},
            ),
            CandidateFeatureSet(
                "promotions", pd.DataFrame({"promo": extras["promo"]}, index=idx), ("price_promotion",), availability={"promo": "available"}
            ),
            CandidateFeatureSet(
                "fuel_prices", pd.DataFrame({"fuel": extras["noise"]}, index=idx), ("economy", "fuel"), availability={"fuel": "available"}
            ),
            CandidateFeatureSet(
                "final_invoice",
                pd.DataFrame({"final_invoice": df.sales * 1.01}, index=idx),
                ("events",),
                availability={"final_invoice": "unavailable"},
            ),
        ]
    )
    cfg = fast_config(
        tmp_path,
        validation=ValidationConfig(
            strategy="rolling_origin", timestamp_column="week", group_columns=["store_id"], min_train_periods=40, test_periods=10, step=10
        ),
        autonomous_rounds=4,
        minimum_feature_gain=0.5,
    )
    s = AutonomousAutoMLScientist(cfg, context="weekly store ice cream sales demand", candidate_source=source)
    r = s.fit(df, "sales")
    loop = r.loop
    assert loop is not None
    assert loop.iterations >= 1
    assert set(loop.kept_candidates) == {"weather_daily", "promotions"}
    assert {"temperature", "promo"} <= set(loop.final_feature_columns) and "final_invoice" not in loop.final_feature_columns
    decisions = dict(zip(loop.audit_trail.candidate, loop.audit_trail.decision, strict=True))
    assert decisions["fuel_prices"] in {REJECT, INCONCLUSIVE}
    assert decisions["final_invoice"] == INVALID
    assert loop.stop_reason in {NO_MEANINGFUL_IMPROVEMENT, NO_CANDIDATES, MAX_ITERATIONS}
    assert loop.model_card["stop_reason"] == loop.stop_reason
    hold = loop.scientist.automl.holdouts["sales"]
    assert len(hold.evaluations) == 1 and loop.holdout_evaluation.score < 10  # final system evaluated once, and it generalizes
    assert loop.baseline_history.baseline_mean.iloc[-1] < loop.baseline_history.baseline_mean.iloc[0]
    assert (tmp_path / "experiment_memory.json").exists() and (tmp_path / "model_card.json").exists()
    saved = json.loads((tmp_path / "experiment_memory.json").read_text())
    assert {d["decision"] for d in saved} >= {KEEP, INVALID}
    assert all(rec.hypothesis and rec.evidence for rec in loop.memory.records)


def test_stop_conditions_record_reason(tmp_path):
    df, extras = panel(n_weeks=60, n_stores=2, seed=2)
    source = InMemoryCandidateSource(
        [
            CandidateFeatureSet(
                "weather_daily",
                pd.DataFrame({"temperature": extras["temp"]}, index=df.index),
                ("weather",),
                availability={"temperature": "available"},
            )
        ]
    )
    cfg = fast_config(tmp_path, evaluate_holdout=False)
    orch = PredictiveDiscoveryOrchestrator(cfg, candidate_source=source, stop=StopConfig(max_iterations=0))
    out = orch.run(df, "sales", "weekly sales")
    assert out.stop_reason == MAX_ITERATIONS and out.iterations == 0
    orch2 = PredictiveDiscoveryOrchestrator(
        fast_config(tmp_path / "b", evaluate_holdout=False), candidate_source=source, human_stop=lambda: True
    )
    assert orch2.run(df, "sales", "weekly sales").stop_reason == "HUMAN_STOP"


# --------------------------------------------------------------------------------------------- #
# Value of Information consumes validated uplift only
# --------------------------------------------------------------------------------------------- #
def test_value_of_information_refuses_unestablished_uplift():
    judge = ExperimentJudge()
    engine = ValueOfInformationEngine(higher_is_better=False)
    business = BusinessValueModel(value_per_error_unit=1000, annual_decisions=100, realization_rate=0.5)
    cost = DataCost(annual_license_cost=100_000, one_time_integration_cost=50_000)
    huge_but_invalid = judge.evaluate([100] * 5, [10] * 5, higher_is_better=False, invalid_reasons=["TEMPORAL_LEAKAGE"])
    res = engine.evaluate_experiment(huge_but_invalid, business, cost, candidate="leak", metric="rmse")
    assert res.recommendation.startswith("NOT_ESTABLISHED") and res.expected_annual_business_value == 0
    assert res.evidence_decision == INVALID
    with pytest.raises(UpliftNotEstablished):
        engine.evaluate_experiment(huge_but_invalid, business, cost, strict=True)
    inconclusive = judge.evaluate([114, 110, 118, 109, 115], [110, 113, 113, 111, 113], higher_is_better=False)
    assert engine.evaluate_experiment(inconclusive, business, cost).recommendation == f"NOT_ESTABLISHED_{INCONCLUSIVE}"
    kept = judge.evaluate([100, 101, 99, 100, 100], [80, 82, 78, 81, 79], higher_is_better=False)
    assert kept.decision == KEEP
    ok = engine.evaluate_experiment(kept, business, cost, candidate="weather", metric="rmse")
    assert ok.recommendation == "ACQUIRE_OR_KEEP" and ok.evidence_decision == KEEP
    assert ok.improvement_absolute == pytest.approx(kept.uplift.mean)
    assert ok.expected_annual_business_value == pytest.approx(kept.uplift.mean * 1000 * 100 * 0.5)
    assert 0 < ok.conservative_annual_business_value <= ok.expected_annual_business_value
    voi = pd.DataFrame(
        [
            {"candidate": "weather", "first_year_cost": 100, "first_year_net_value": 500, "evidence_decision": KEEP},
            {"candidate": "leak", "first_year_cost": 10, "first_year_net_value": 9_000, "evidence_decision": INVALID},
            {"candidate": "maybe", "first_year_cost": 10, "first_year_net_value": 900, "evidence_decision": INCONCLUSIVE},
        ]
    )
    port = optimize_information_portfolio(voi, annual_budget=200)
    assert port["selected"] == ["weather"] and set(port["excluded"]) == {"leak", "maybe"}


# --------------------------------------------------------------------------------------------- #
# Configuration is wired, diagnostics are visible
# --------------------------------------------------------------------------------------------- #
def test_configuration_flags_are_consumed(tmp_path):
    df, extras = panel(n_weeks=60, n_stores=2)
    df = df.assign(temp=extras["temp"], promo=extras["promo"])
    cfg = fast_config(
        tmp_path, run_feature_ablation=True, max_ablation_features=2, run_noise_controls=True, noise_controls=3, minimum_feature_gain=0.2
    )
    r = AutonomousAutoMLScientist(cfg, "weekly sales", feature_families={"weather": ["temp"], "promo": ["promo"]}).fit(df, "sales")
    assert r.ablation is not None and len(r.ablation) == 2
    fam = r.information_value.set_index("feature_family")
    assert fam.noise_threshold.notna().all() and (fam.required_gain >= 0.2).all()
    assert fam.loc["weather", "decision"] == KEEP
    assert r.uncertainty["calibration"].startswith("random")
    assert (tmp_path / "diagnostics.csv").exists()
    assert not r.diagnostics.has_errors


def test_conformal_chronological_calibration_uses_last_rows():
    X = pd.DataFrame({"x": np.linspace(0, 10, 100)})
    y = 2 * X.x
    c = SplitConformalRegressor(Pipeline([("m", Ridge())]), alpha=0.1, chronological=True).fit(X, y)
    assert c.radius_ >= 0
    assert len(c.predict(X.iloc[:3]).lower) == 3


def test_classification_uses_stratified_folds_and_locked_holdout(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(label=y)
    cfg = fast_config(tmp_path, task="classification", metric="roc_auc", models=("logistic",))
    r = AutoML(cfg).fit(df, "label")
    assert r.folds["label"].strategy_name == "stratified_kfold"
    assert r.best_results.holdout_performance.iloc[0] > 0.8
    assert len(r.holdouts["label"].evaluations) == 1
