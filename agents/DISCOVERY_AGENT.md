# Data Discovery Agent

Status: **IMPLEMENTED** for metadata-first discovery with lexical relevance, join-path reasoning, availability metadata and bounded profiling (`automl_py.discovery_agent`). Embedding similarity and LLM reasoning are **optional hooks** (`EmbeddingRelevanceProvider`, `HypothesisReasoner`), not required by the deterministic core.

## Mission
Find governed information that could plausibly address a predictive hypothesis and return safe, ranked candidate datasets and join paths. Relevance identifies candidates; it never proves predictive value.

## Search order
1. Existing enterprise semantic/ontology objects.
2. Derivable information from governed data.
3. Approved feature stores/catalogs.
4. Approved external catalogs/marketplaces (`ExternalSignalScout`, human-governed).
5. Return UNKNOWN when no defensible candidate exists.

## Relevance (hybrid, deterministic guardrail)
```text
lexical relevance (guardrail) + embedding similarity (optional) + graph distance
+ grain compatibility + joinability + experiment-memory prior -> candidate relevance
```

## Responsibilities
- normalize Snowflake/Databricks metadata into `SemanticGraph`;
- traverse semantic relationships and preserve join provenance;
- derive point-in-time availability from catalog metadata (`available` / `unavailable` / `unknown`; never hard-coded);
- perform bounded profiling of only the top candidates;
- flag row-explosion and low-coverage risk (`JoinValidator` → INVALID with reason codes).

## Permission levels (defaults)
`DISCOVER` metadata only → `PROFILE` bounded approved queries → `EXPERIMENT` approved extraction → `ACQUIRE` human-controlled. Read-only, RBAC-respecting, bounded queries, no DDL/writes/deletes/purchases.

## Must not
- invent joins unsupported by evidence;
- bypass RBAC;
- issue mutations/DDL;
- perform unbounded production scans;
- treat semantic relevance as proof of predictive value;
- purchase data.
