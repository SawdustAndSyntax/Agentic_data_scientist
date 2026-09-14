"""Tests for the second sprint: corrected statistics, search budget, candidate sources,
point-in-time joins, the report, and the optional reasoner."""

from __future__ import annotations

import json
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from automl_py import (
    INVALID,
    KEEP,
    REJECT,
    AutoML,
    AutoMLConfig,
    AutonomousAutoMLScientist,
    CandidateData,
    CandidateFeatureSet,
    CompositeCandidateSource,
    DataDiscoveryAgent,
    DerivedFeatureSource,
    Diagnosis,
    DiscoveryCandidateSource,
    HypothesisGenerator,
    InMemoryCandidateSource,
    InMemoryCatalogAdapter,
    PointInTimeJoiner,
    PredictiveDiscoveryOrchestrator,
    SemanticEntity,
    SemanticField,
    SemanticRelationship,
    StopConfig,
    ValidationConfig,
    build_manifest,
    key_join,
    render_report,
)
from automl_py.llm import AnthropicHypothesisReasoner
from automl_py.orchestrator import LoopState

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
        "noise_controls": 3,
        "artifact_dir": str(tmp_path),
    }
    base.update(kw)
    return AutoMLConfig(**base)


def panel(n_weeks=100, n_stores=3, seed=0):
    rng = np.random.default_rng(seed)
    weeks = np.repeat(np.arange(1, n_weeks + 1), n_stores)
    stores = np.tile(np.arange(1, n_stores + 1), n_weeks)
    n = len(weeks)
    x = rng.normal(size=n)
    temp = 20 + 10 * np.sin(2 * np.pi * weeks / 52) + rng.normal(0, 2, n)
    y = 50 + 3 * x + 2.5 * temp + rng.normal(0, 2, n)
    return pd.DataFrame({"week": weeks, "store_id": stores, "x": x, "sales": y}), temp


# --------------------------------------------------------------------------------------------- #
# Search budget
# --------------------------------------------------------------------------------------------- #
def test_screening_limits_full_validation_and_time_budget_is_recorded(tmp_path):
    df, _ = panel(60, 2)
    cfg = fast_config(
        tmp_path,
        preprocessors=("original", "scale"),
        imputation_strategies=("median", "mean"),
        models=("ridge", "tree", "knn"),
        screen_top_k=3,
    )
    r = AutoML(cfg).fit(df, "sales")
    stages = r.results.stage.value_counts()
    assert stages["full"] == 3 and stages["screened_out"] == 9
    assert r.results[r.results.stage == "screened_out"].screen_score.notna().all()
    assert r.best_row("sales").stage == "full"
    cfg2 = fast_config(tmp_path / "b", models=("ridge", "tree", "knn"), screen_configurations=False, max_search_seconds=0.0)
    r2 = AutoML(cfg2).fit(df, "sales")
    assert (r2.results.stage == "full").sum() == 1 and (r2.results.stage == "skipped_time_budget").sum() == 2
    assert any("search budget" in e.message for e in r2.diagnostics.events)


# --------------------------------------------------------------------------------------------- #
# Point-in-time joins and candidate sources
# --------------------------------------------------------------------------------------------- #
def test_point_in_time_join_respects_availability_lag():
    left = pd.DataFrame({"store": [1, 1, 1, 2], "week": [1, 2, 3, 2]}, index=[10, 11, 12, 13])
    right = pd.DataFrame({"store": [1, 1, 1, 2], "week": [1, 2, 3, 2], "temp": [10.0, 20.0, 30.0, 99.0]})
    j = PointInTimeJoiner()
    same = j.join(left, right, left_time="week", right_time="week", left_keys=["store"], right_keys=["store"])
    assert list(same.index) == [10, 11, 12, 13] and same.temp.tolist() == [10.0, 20.0, 30.0, 99.0]
    lagged = j.join(left, right, left_time="week", right_time="week", left_keys=["store"], right_keys=["store"], availability_lag=1)
    # a value observed in week t is only usable from week t+1: week 1 has nothing, week 2 sees week 1's value
    assert np.isnan(lagged.temp.iloc[0]) and lagged.temp.tolist()[1:3] == [10.0, 20.0] and np.isnan(lagged.temp.iloc[3])
    exploded = key_join(left, pd.DataFrame({"store": [1, 1], "v": [1, 2]}), ["store"], ["store"])
    assert exploded.index.has_duplicates


def test_derived_feature_source_builds_transforms_interactions_and_history():
    df, _ = panel(30, 2)
    X = df[["store_id", "x"]].copy()
    X["z"] = np.arange(len(X)) * 0.1
    state = LoopState(0, list(X.columns), X, df.sales, [1.0], [], timestamps=df.week, groups=df.store_id, frame=df, target="sales")
    hyp = HypothesisGenerator().generate(
        Diagnosis(
            target="sales",
            metric="rmse",
            baseline_scores=[1.0],
            current_columns=list(X.columns),
            residual_diagnostics=pd.DataFrame([{"feature": "x", "residual_correlation": 0.4, "residual_mutual_information": 0.2}]),
        )
    )
    derivable = next(h for h in hyp if h.kind == "derivable")
    cands = DerivedFeatureSource().candidates(derivable, state)
    names = {c.name for c in cands}
    assert "derived:x:nonlinear" in names and "derived:x:history" in names and any(n.startswith("derived:xx") for n in names)
    hist = next(c for c in cands if c.name == "derived:x:history").frame
    assert list(hist.index) == list(X.index)
    # lag never uses the current or a future period: within a store, lag1 at week w equals x at week w-1
    s1 = df[df.store_id == 1].sort_values("week")
    assert np.isnan(hist.loc[s1.index[0], "x__lag1"]) and hist.loc[s1.index[1], "x__lag1"] == pytest.approx(s1.x.iloc[0])
    assert all(v == "available" for c in cands for v in c.availability.values())
    assert DerivedFeatureSource().candidates(next(h for h in hyp if h.kind == "external"), state) == []


def catalog_with_weather(df, temp):
    sales = SemanticEntity("sales", "a.sales", "memory", "table", "weekly store sales", [SemanticField("store_id", role="key")])
    weather = SemanticEntity(
        "weather",
        "ext.weather",
        "memory",
        "table",
        "daily weather temperature humidity",
        [SemanticField("store_id", role="key"), SemanticField("temperature")],
        metadata={"availability_offset": "-1D"},
    )
    invoice = SemanticEntity(
        "invoice", "erp.final_invoice", "memory", "table", "final invoice temperature revenue", metadata={"availability_offset": "+14D"}
    )
    rels = [
        SemanticRelationship("s_w", "a.sales", "ext.weather", ("store_id",), ("store_id",)),
        SemanticRelationship("s_i", "a.sales", "erp.final_invoice", ("store_id",), ("store_id",)),
    ]
    frames = {
        "ext.weather": pd.DataFrame({"store_id": df.store_id, "week": df.week, "temperature": temp}),
        "erp.final_invoice": pd.DataFrame({"store_id": df.store_id, "week": df.week, "final_invoice": df.sales * 1.01}),
    }
    return DataDiscoveryAgent(InMemoryCatalogAdapter([sales, weather, invoice], rels, frames)), frames


def test_discovery_backed_source_runs_inside_the_loop(tmp_path):
    df, temp = panel(120, 4, seed=3)
    agent, frames = catalog_with_weather(df, temp)

    def loader(candidate):
        frame = frames.get(candidate.entity.qualified_name)
        if frame is None:
            return None
        return CandidateData(frame, left_keys=["store_id"], right_keys=["store_id"], right_time="week", availability_lag=0, cost=85_000)

    source = DiscoveryCandidateSource(
        agent, loader, anchor_entities=["a.sales"], timestamp_column="week", request_context="weekly store demand"
    )
    cfg = fast_config(
        tmp_path,
        evaluate_holdout=False,
        minimum_feature_gain=0.5,
        validation=ValidationConfig(
            strategy="rolling_origin", timestamp_column="week", group_columns=["store_id"], min_train_periods=40, test_periods=10, step=10
        ),
    )
    out = PredictiveDiscoveryOrchestrator(cfg, candidate_source=source, stop=StopConfig(max_iterations=3)).run(
        df, "sales", "weekly store ice cream sales demand"
    )
    decisions = dict(zip(out.audit_trail.candidate, out.audit_trail.decision, strict=True))
    assert decisions["ext.weather"] == KEEP and "temperature" in out.final_feature_columns
    assert decisions["erp.final_invoice"] == INVALID  # post-outcome per catalog metadata, never trained
    rec = next(r for r in out.memory.records if r.candidate == "ext.weather")
    assert rec.join_path == ["a.sales", "ext.weather"] and rec.extra["source"] == "discovery:memory" and rec.extra["cost"] == 85_000
    assert source.last_report is not None and "availability" in source.last_report.columns
    assert len(out.scientist.automl.holdouts["sales"].evaluations) == 1


def test_forward_selection_adopts_best_survivor_and_retests_rest(tmp_path):
    df, temp = panel(120, 4, seed=5)
    rng = np.random.default_rng(1)
    promo = rng.binomial(1, 0.3, len(df))
    df["sales"] = df.sales + 15 * promo
    source = InMemoryCandidateSource(
        [
            CandidateFeatureSet(
                "weather_daily",
                pd.DataFrame({"temperature": temp}, index=df.index),
                ("weather",),
                availability={"temperature": "available"},
            ),
            CandidateFeatureSet(
                "promotions", pd.DataFrame({"promo": promo}, index=df.index), ("price_promotion",), availability={"promo": "available"}
            ),
        ]
    )
    cfg = fast_config(tmp_path, evaluate_holdout=False, minimum_feature_gain=0.5, adopt_per_iteration=1)
    out = PredictiveDiscoveryOrchestrator(cfg, candidate_source=source, stop=StopConfig(max_iterations=4)).run(
        df, "sales", "weekly store sales demand"
    )
    trail = out.audit_trail
    first = trail[trail.iteration == 1]
    assert set(first.candidate) == {"weather_daily", "promotions"} and (first.decision == KEEP).all()
    assert first.extra.apply(lambda e: e["adopted"]).sum() == 1  # one adoption per iteration
    assert set(out.kept_candidates) == {"weather_daily", "promotions"} and out.iterations >= 2
    assert (trail.candidate == "promotions").sum() + (trail.candidate == "weather_daily").sum() == 3  # the runner-up was re-tested once
    assert out.baseline_history.baseline_mean.is_monotonic_decreasing


def test_composite_source_deduplicates():
    a = InMemoryCandidateSource([CandidateFeatureSet("w", pd.DataFrame({"t": [1.0]}), ("weather",))])
    b = InMemoryCandidateSource(
        [
            CandidateFeatureSet("w", pd.DataFrame({"t": [1.0]}), ("weather",)),
            CandidateFeatureSet("v", pd.DataFrame({"u": [1.0]}), ("weather",)),
        ]
    )
    hyp = HypothesisGenerator().generate(
        Diagnosis(target="y", metric="rmse", baseline_scores=[1], current_columns=["x"], context="sales demand")
    )
    weather = next(h for h in hyp if h.concept == "weather")
    state = LoopState(0, ["x"], pd.DataFrame({"x": [1.0]}), pd.Series([1.0]), [1.0], [])
    assert [c.name for c in CompositeCandidateSource([a, b]).candidates(weather, state)] == ["w", "v"]


# --------------------------------------------------------------------------------------------- #
# Report and manifest
# --------------------------------------------------------------------------------------------- #
def test_report_and_manifest_are_written_and_complete(tmp_path):
    df, temp = panel(80, 3, seed=2)
    source = InMemoryCandidateSource(
        [
            CandidateFeatureSet(
                "weather_daily",
                pd.DataFrame({"temperature": temp}, index=df.index),
                ("weather",),
                availability={"temperature": "available"},
            )
        ]
    )
    cfg = fast_config(tmp_path, minimum_feature_gain=0.5, autonomous_rounds=2)
    r = AutonomousAutoMLScientist(cfg, "weekly store sales demand", candidate_source=source).fit(df, "sales")
    text = (tmp_path / "report.md").read_text()
    for section in (
        "## Objective and validation design",
        "## Baseline champion",
        "## Autonomous loop",
        "### Experiments",
        "## Final system",
        "## Reproducibility",
    ):
        assert section in text
    assert "weather_daily" in text and "KEEP" in text and "evaluated 1 time" in text
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["data"]["rows"] == len(df) and manifest["packages"]["scikit-learn"] != "unknown" and manifest["random_state"] == 100
    m2 = build_manifest(cfg, df)
    assert m2["data"]["fingerprint"] == manifest["data"]["fingerprint"]
    assert render_report(r.scientist).startswith("# Predictive Discovery report")


# --------------------------------------------------------------------------------------------- #
# Optional reasoner with a fake client
# --------------------------------------------------------------------------------------------- #
class FakeClient:
    def __init__(self, reply: str, stop_reason: str = "end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[types.SimpleNamespace(type="text", text=self.reply)])


def diagnosis():
    return Diagnosis(
        target="sales", metric="rmse", baseline_scores=[10.0, 11.0], current_columns=["x", "price"], context="weekly store sales demand"
    )


def test_reasoner_only_accepts_hypotheses_that_cite_facts():
    gen = HypothesisGenerator()
    base = gen.generate(diagnosis())
    reply = json.dumps(
        {
            "rerank": [base[-1].hypothesis_id, base[0].hypothesis_id],
            "new": [
                {
                    "statement": "School holidays shift demand.",
                    "evidence_ids": ["F2"],
                    "search_concepts": ["school calendar"],
                    "kind": "external",
                    "confidence": 0.95,
                },
                {
                    "statement": "Fabricated claim.",
                    "evidence_ids": ["F99"],
                    "search_concepts": ["nothing"],
                    "kind": "external",
                    "confidence": 0.9,
                },
                {"statement": "No concepts.", "evidence_ids": ["F1"], "search_concepts": [], "kind": "external"},
            ],
        }
    )
    client = FakeClient(reply)
    out = AnthropicHypothesisReasoner(client, model="claude-opus-5").refine(list(base), diagnosis())
    assert out[0].hypothesis_id == base[-1].hypothesis_id and out[1].hypothesis_id == base[0].hypothesis_id
    llm = [h for h in out if h.hypothesis_id.startswith("H-LLM")]
    assert len(llm) == 1 and llm[0].confidence == 0.7 and llm[0].evidence[0].startswith("F2:") and llm[0].kind == "external"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5" and call["thinking"] == {"type": "adaptive"} and '"FACTS"' in call["messages"][0]["content"]


def test_reasoner_leaves_hypotheses_unchanged_on_refusal_or_garbage():
    gen = HypothesisGenerator()
    base = gen.generate(diagnosis())
    assert AnthropicHypothesisReasoner(FakeClient("", stop_reason="refusal")).refine(list(base), diagnosis()) == base
    assert AnthropicHypothesisReasoner(FakeClient("not json at all")).refine(list(base), diagnosis()) == base

    class Boom:
        messages = types.SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("network")))

    assert AnthropicHypothesisReasoner(Boom()).refine(list(base), diagnosis()) == base
    # the generator hook integrates it and still never marks anything KEEP
    hyps = HypothesisGenerator(reasoner=AnthropicHypothesisReasoner(FakeClient(json.dumps({"rerank": [], "new": []})))).generate(
        diagnosis()
    )
    assert hyps and all(h.confidence <= 0.9 for h in hyps)


def test_rejected_candidates_never_become_keep_via_reasoner(tmp_path):
    df, _temp = panel(60, 2, seed=4)
    source = InMemoryCandidateSource(
        [
            CandidateFeatureSet(
                "noise_feed",
                pd.DataFrame({"n": np.random.default_rng(0).normal(size=len(df))}, index=df.index),
                ("weather",),
                availability={"n": "available"},
            )
        ]
    )
    reasoner = AnthropicHypothesisReasoner(
        FakeClient(
            json.dumps(
                {
                    "rerank": [],
                    "new": [
                        {
                            "statement": "Noise is great",
                            "evidence_ids": ["F1"],
                            "search_concepts": ["weather"],
                            "kind": "external",
                            "confidence": 1.0,
                        }
                    ],
                }
            )
        )
    )
    cfg = fast_config(tmp_path, minimum_feature_gain=0.5, autonomous_rounds=2)
    r = AutonomousAutoMLScientist(cfg, "weekly sales demand", candidate_source=source, reasoner=reasoner).fit(df, "sales")
    assert r.loop.kept_candidates == [] and set(r.loop.audit_trail.decision) <= {REJECT, "INCONCLUSIVE"}
    assert any(h.startswith("H-LLM") for h in r.loop.hypotheses_log.hypothesis_id)
