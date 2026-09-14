from __future__ import annotations

import ast
import json
import re

import pandas as pd

from ..contracts import SemanticEntity, SemanticField, SemanticRelationship
from .base import CatalogAdapter
from .sql import DBAPIExecutor

SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def qi(x: str) -> str:
    if SAFE_IDENT.match(x):
        return x
    return '"' + x.replace('"', '""') + '"'


def arr(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(map(str, v))
    s = str(v)
    for parser in (json.loads, ast.literal_eval):
        try:
            z = parser(s)
            return tuple(map(str, z)) if isinstance(z, (list, tuple)) else (str(z),)
        except Exception:
            pass
    return tuple(x.strip().strip("\"'") for x in s.strip("[]").split(",") if x.strip())


class SnowflakeCatalogAdapter(CatalogAdapter):
    platform = "snowflake"

    def __init__(self, connection=None, *, executor=None, database: str, schemas: list[str] | None = None):
        self.exec = executor or DBAPIExecutor(connection)
        self.database = database
        self.schemas = schemas

    def _where_schema(self, col="table_schema"):
        if not self.schemas:
            return ""
        vals = ",".join("'" + s.replace("'", "''") + "'" for s in self.schemas)
        return f" WHERE {col} IN ({vals})"

    def entities(self):
        db = qi(self.database)
        entities = {}
        tables = self.exec.query(
            f"SELECT table_catalog,table_schema,table_name,table_type,comment FROM {db}.information_schema.tables" + self._where_schema()
        )
        cols = self.exec.query(
            f"SELECT table_catalog,table_schema,table_name,column_name,data_type,comment FROM {db}.information_schema.columns"
            + self._where_schema()
        )
        for _, r in tables.iterrows():
            q = f"{r.table_catalog}.{r.table_schema}.{r.table_name}"
            entities[q] = SemanticEntity(str(r.table_name), q, self.platform, str(r.table_type).lower(), str(r.get("comment") or ""))
        for _, r in cols.iterrows():
            q = f"{r.table_catalog}.{r.table_schema}.{r.table_name}"
            if q in entities:
                entities[q].fields.append(SemanticField(str(r.column_name), str(r.data_type), description=str(r.get("comment") or "")))
        # Snowflake-native semantic view objects augment physical metadata.
        try:
            st = self.exec.query(
                f"SELECT semantic_view_catalog,semantic_view_schema,semantic_view_name,name,base_table_catalog,base_table_schema,base_table_name,primary_keys,comment FROM {db}.information_schema.semantic_tables"
            )
            dims = self.exec.query(
                f"SELECT semantic_view_catalog,semantic_view_schema,semantic_view_name,table_name,name,data_type,expression,synonyms,comment FROM {db}.information_schema.semantic_dimensions"
            )
            mets = self.exec.query(
                f"SELECT semantic_view_catalog,semantic_view_schema,semantic_view_name,table_name,name,data_type,expression,synonyms,comment FROM {db}.information_schema.semantic_metrics"
            )
            for _, r in st.iterrows():
                q = f"semantic:{r.semantic_view_catalog}.{r.semantic_view_schema}.{r.semantic_view_name}.{r['name']}"
                entities[q] = SemanticEntity(
                    str(r["name"]),
                    q,
                    self.platform,
                    "semantic_table",
                    str(r.get("comment") or ""),
                    primary_keys=arr(r.get("primary_keys")),
                    metadata={"base_table": f"{r.base_table_catalog}.{r.base_table_schema}.{r.base_table_name}"},
                )
            for frame, role in ((dims, "dimension"), (mets, "metric")):
                for _, r in frame.iterrows():
                    q = f"semantic:{r.semantic_view_catalog}.{r.semantic_view_schema}.{r.semantic_view_name}.{r.table_name}"
                    if q in entities:
                        entities[q].fields.append(
                            SemanticField(
                                str(r["name"]),
                                str(r.data_type),
                                role,
                                str(r.get("comment") or ""),
                                str(r.get("expression") or ""),
                                arr(r.get("synonyms")),
                            )
                        )
        except Exception:
            pass
        return list(entities.values())

    def relationships(self):
        db = qi(self.database)
        out = []
        try:
            rels = self.exec.query(
                f"SELECT semantic_view_catalog,semantic_view_schema,semantic_view_name,name,table_name,foreign_keys,ref_table_name,ref_keys FROM {db}.information_schema.semantic_relationships"
            )
            for _, r in rels.iterrows():
                prefix = f"semantic:{r.semantic_view_catalog}.{r.semantic_view_schema}.{r.semantic_view_name}."
                out.append(
                    SemanticRelationship(
                        str(r["name"]),
                        prefix + str(r.table_name),
                        prefix + str(r.ref_table_name),
                        arr(r.foreign_keys),
                        arr(r.ref_keys),
                        "semantic",
                        1.0,
                    )
                )
        except Exception:
            pass
        return out

    def sample(self, qualified_name, limit=1000):
        if qualified_name.startswith("semantic:"):
            raise ValueError("Sample the semantic table base_table from entity metadata, not the semantic logical name.")
        parts = qualified_name.split(".")
        if len(parts) != 3:
            raise ValueError("Expected database.schema.table")
        return self.exec.query(f"SELECT * FROM {'.'.join(qi(p) for p in parts)} LIMIT {int(limit)}")

    def profile(self, qualified_name, sample_rows=1000):
        df = self.sample(qualified_name, sample_rows)
        coverage = float(1 - df.isna().mean().mean()) if not df.empty else 0.0
        return {"qualified_name": qualified_name, "sample_rows": len(df), "coverage": coverage, "quality": coverage}
