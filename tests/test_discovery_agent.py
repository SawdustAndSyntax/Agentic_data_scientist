import pandas as pd

from automl_py.discovery_agent import (
    DataDiscoveryAgent,
    DiscoveryRequest,
    InMemoryCatalogAdapter,
    JoinValidator,
    PredictiveDiscoveryLoop,
    SemanticEntity,
    SemanticField,
    SemanticRelationship,
    SnowflakeCatalogAdapter,
)


def build_catalog():
    sales = SemanticEntity(
        "sales",
        "analytics.public.sales",
        "memory",
        "table",
        "weekly store ice cream sales",
        [SemanticField("store_id", "int", "key"), SemanticField("week", "date", "dimension"), SemanticField("sales", "float", "metric")],
    )
    store = SemanticEntity(
        "store",
        "analytics.public.store",
        "memory",
        "table",
        "store master geography",
        [SemanticField("store_id", "int", "key"), SemanticField("weather_station_id", "int", "key")],
    )
    weather = SemanticEntity(
        "weather_daily",
        "external.weather.daily",
        "memory",
        "table",
        "daily weather temperature humidity precipitation",
        [
            SemanticField("weather_station_id", "int", "key"),
            SemanticField("temperature", "float", "dimension"),
            SemanticField("humidity", "float", "dimension"),
        ],
    )
    noise = SemanticEntity("hr_payroll", "corp.hr.payroll", "memory", "table", "employee payroll")
    rels = [
        SemanticRelationship("sales_store", "analytics.public.sales", "analytics.public.store", ("store_id",), ("store_id",)),
        SemanticRelationship(
            "store_weather", "analytics.public.store", "external.weather.daily", ("weather_station_id",), ("weather_station_id",)
        ),
    ]
    frames = {
        "external.weather.daily": pd.DataFrame(
            {"weather_station_id": [10, 20, 30], "temperature": [70, 80, 75], "humidity": [0.4, 0.6, 0.5]}
        ),
        "analytics.public.sales": pd.DataFrame({"store_id": [1, 2, 3], "sales": [10, 20, 30]}),
        "analytics.public.store": pd.DataFrame({"store_id": [1, 2, 3], "weather_station_id": [10, 20, 30]}),
    }
    return InMemoryCatalogAdapter([sales, store, weather, noise], rels, frames)


def test_discovery_finds_weather_through_graph():
    agent = DataDiscoveryAgent(build_catalog())
    req = DiscoveryRequest(target="ice cream sales", hypothesis="weather temperature", context="weekly store demand", top_k=5)
    out = agent.discover(req, anchor_entities=["analytics.public.sales"])
    assert out
    assert out[0].entity.qualified_name == "external.weather.daily"
    assert out[0].join_plan is not None
    assert out[0].join_plan.path == ["analytics.public.sales", "analytics.public.store", "external.weather.daily"]


def test_join_validator_rejects_row_explosion():
    left = pd.DataFrame({"id": [1, 2, 3], "y": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 1, 1, 2, 3], "x": [1, 2, 3, 4, 5]})
    v = JoinValidator().validate_frames(left, right, ["id"], ["id"])
    assert v.status == "review"
    assert v.row_multiplier > 1.2
    assert v.duplicate_key_risk


def test_predictive_discovery_loop_keeps_uplift():
    adapter = build_catalog()
    agent = DataDiscoveryAgent(adapter)
    base = pd.DataFrame({"store_id": [1, 2, 3], "sales": [10, 20, 30]})
    req = DiscoveryRequest(target="sales", hypothesis="weather temperature", context="ice cream demand", top_k=3)

    def loader(candidate):
        # For this unit test, map the top weather candidate directly by store id.
        extra = pd.DataFrame({"store_id": [1, 2, 3], "temperature": [70, 80, 75]})
        return extra, ["store_id"], ["store_id"]

    def runner(df):
        return {"score": 0.8 if "temperature" in df.columns else 0.6, "higher_is_better": True}

    r = PredictiveDiscoveryLoop(agent).run(
        req, anchor_entities=["analytics.public.sales"], base_df=base, candidate_loader=loader, experiment_runner=runner, top_n=1
    )
    assert r.tested_candidates.iloc[0].status == "keep"
    assert r.tested_candidates.iloc[0].uplift > 0


class FakeExecutor:
    def query(self, sql):
        q = sql.lower()
        if "semantic_tables" in q:
            return pd.DataFrame(
                [
                    {
                        "semantic_view_catalog": "ANALYTICS",
                        "semantic_view_schema": "PUBLIC",
                        "semantic_view_name": "RETAIL",
                        "name": "STORE",
                        "base_table_catalog": "ANALYTICS",
                        "base_table_schema": "PUBLIC",
                        "base_table_name": "STORE",
                        "primary_keys": ["STORE_ID"],
                        "comment": "store entity",
                    },
                    {
                        "semantic_view_catalog": "ANALYTICS",
                        "semantic_view_schema": "PUBLIC",
                        "semantic_view_name": "RETAIL",
                        "name": "WEATHER",
                        "base_table_catalog": "ANALYTICS",
                        "base_table_schema": "PUBLIC",
                        "base_table_name": "WEATHER_DAILY",
                        "primary_keys": ["STATION_ID"],
                        "comment": "weather observations",
                    },
                ]
            )
        if "semantic_dimensions" in q:
            return pd.DataFrame(
                [
                    {
                        "semantic_view_catalog": "ANALYTICS",
                        "semantic_view_schema": "PUBLIC",
                        "semantic_view_name": "RETAIL",
                        "table_name": "WEATHER",
                        "name": "TEMPERATURE",
                        "data_type": "NUMBER",
                        "expression": "temperature",
                        "synonyms": ["TEMP"],
                        "comment": "air temperature",
                    }
                ]
            )
        if "semantic_metrics" in q:
            return pd.DataFrame(
                columns=[
                    "semantic_view_catalog",
                    "semantic_view_schema",
                    "semantic_view_name",
                    "table_name",
                    "name",
                    "data_type",
                    "expression",
                    "synonyms",
                    "comment",
                ]
            )
        if "semantic_relationships" in q:
            return pd.DataFrame(
                [
                    {
                        "semantic_view_catalog": "ANALYTICS",
                        "semantic_view_schema": "PUBLIC",
                        "semantic_view_name": "RETAIL",
                        "name": "STORE_WEATHER",
                        "table_name": "STORE",
                        "foreign_keys": ["WEATHER_STATION_ID"],
                        "ref_table_name": "WEATHER",
                        "ref_keys": ["STATION_ID"],
                    }
                ]
            )
        if "information_schema.tables" in q:
            return pd.DataFrame(columns=["table_catalog", "table_schema", "table_name", "table_type", "comment"])
        if "information_schema.columns" in q:
            return pd.DataFrame(columns=["table_catalog", "table_schema", "table_name", "column_name", "data_type", "comment"])
        if "select * from" in q:
            return pd.DataFrame({"x": [1, 2, 3]})
        raise AssertionError(sql)


def test_snowflake_semantic_adapter():
    a = SnowflakeCatalogAdapter(executor=FakeExecutor(), database="ANALYTICS")
    entities = a.entities()
    rels = a.relationships()
    weather = next(e for e in entities if e.name == "WEATHER")
    assert any(f.name == "TEMPERATURE" and f.role == "dimension" for f in weather.fields)
    assert rels[0].source.endswith(".STORE") and rels[0].target.endswith(".WEATHER")
    assert rels[0].source_keys == ("WEATHER_STATION_ID",)
