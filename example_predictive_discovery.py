from automl_py import (
    DataDiscoveryAgent,
    DiscoveryRequest,
    InMemoryCatalogAdapter,
    SemanticEntity,
    SemanticField,
    SemanticRelationship,
)

sales = SemanticEntity(
    name="sales",
    qualified_name="analytics.public.sales",
    platform="memory",
    description="weekly store product sales",
    fields=[SemanticField("store_id", role="key"), SemanticField("sales", role="metric")],
)
weather = SemanticEntity(
    name="weather_daily",
    qualified_name="external.weather.daily",
    platform="memory",
    description="weather temperature humidity precipitation",
    fields=[SemanticField("store_id", role="key"), SemanticField("temperature")],
)
relationship = SemanticRelationship(
    "sales_weather",
    "analytics.public.sales",
    "external.weather.daily",
    ("store_id",),
    ("store_id",),
)
adapter = InMemoryCatalogAdapter([sales, weather], [relationship])
agent = DataDiscoveryAgent(adapter)

request = DiscoveryRequest(
    target="sales",
    hypothesis="weather temperature",
    context="weekly ice cream demand",
)

print(agent.to_frame(agent.discover(request, anchor_entities=["analytics.public.sales"], profile=False)))
