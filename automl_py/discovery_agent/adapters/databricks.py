from __future__ import annotations

import re

from ..contracts import SemanticEntity, SemanticField, SemanticRelationship
from .base import CatalogAdapter
from .sql import DBAPIExecutor

SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def qi(x):
    return x if SAFE_IDENT.match(x) else "`" + x.replace("`", "``") + "`"


class DatabricksCatalogAdapter(CatalogAdapter):
    platform = "databricks"

    def __init__(self, connection=None, *, executor=None, catalog: str | None = None, schemas: list[str] | None = None):
        self.exec = executor or DBAPIExecutor(connection)
        self.catalog = catalog
        self.schemas = schemas

    def _prefix(self):
        return f"{qi(self.catalog)}.information_schema" if self.catalog else "system.information_schema"

    def _schema_filter(self):
        if not self.schemas:
            return ""
        vals = ",".join("'" + s.replace("'", "''") + "'" for s in self.schemas)
        return f" WHERE table_schema IN ({vals})"

    def entities(self):
        p = self._prefix()
        entities = {}
        tables = self.exec.query(f"SELECT table_catalog,table_schema,table_name,table_type,comment FROM {p}.tables" + self._schema_filter())
        cols = self.exec.query(
            f"SELECT table_catalog,table_schema,table_name,column_name,data_type FROM {p}.columns" + self._schema_filter()
        )
        for _, r in tables.iterrows():
            q = f"{r.table_catalog}.{r.table_schema}.{r.table_name}"
            entities[q] = SemanticEntity(str(r.table_name), q, self.platform, str(r.table_type).lower(), str(r.get("comment") or ""))
        for _, r in cols.iterrows():
            q = f"{r.table_catalog}.{r.table_schema}.{r.table_name}"
            if q in entities:
                entities[q].fields.append(SemanticField(str(r.column_name), str(r.data_type)))
        return list(entities.values())

    def relationships(self):
        # Unity Catalog constraints are privilege-filtered. Query failures are non-fatal
        # because many estates do not declare PK/FK constraints consistently.
        p = self._prefix()
        out = []
        try:
            # Databricks exposes constraint metadata, but reconstructing FK pairs varies
            # by runtime/schema availability; allow an executor implementation to expose
            # a normalized helper view named discovery_relationships when desired.
            rel = self.exec.query(f"SELECT * FROM {p}.referential_constraints")
            # Retain metadata discovery without inventing unsafe joins. Relationships
            # are only emitted when normalized source/target fields exist.
            needed = {
                "source_catalog",
                "source_schema",
                "source_table",
                "source_column",
                "target_catalog",
                "target_schema",
                "target_table",
                "target_column",
            }
            if needed.issubset(set(rel.columns)):
                for _, r in rel.iterrows():
                    s = f"{r.source_catalog}.{r.source_schema}.{r.source_table}"
                    t = f"{r.target_catalog}.{r.target_schema}.{r.target_table}"
                    out.append(
                        SemanticRelationship(
                            str(r.get("constraint_name", "fk")), s, t, (str(r.source_column),), (str(r.target_column),), "foreign_key", 1.0
                        )
                    )
        except Exception:
            pass
        return out

    def sample(self, qualified_name, limit=1000):
        parts = qualified_name.split(".")
        if len(parts) != 3:
            raise ValueError("Expected catalog.schema.table")
        return self.exec.query(f"SELECT * FROM {'.'.join(qi(p) for p in parts)} LIMIT {int(limit)}")

    def profile(self, qualified_name, sample_rows=1000):
        df = self.sample(qualified_name, sample_rows)
        coverage = float(1 - df.isna().mean().mean()) if not df.empty else 0.0
        return {"qualified_name": qualified_name, "sample_rows": len(df), "coverage": coverage, "quality": coverage}
