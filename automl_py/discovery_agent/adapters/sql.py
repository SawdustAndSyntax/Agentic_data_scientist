from __future__ import annotations

import pandas as pd


class DBAPIExecutor:
    """Tiny read-only-ish wrapper around a DB-API connection.

    The adapter only generates SELECT metadata/sample statements. Governance and
    database authorization remain with the supplied connection identity.
    """

    def __init__(self, connection):
        self.connection = connection

    def query(self, sql: str) -> pd.DataFrame:
        cur = self.connection.cursor()
        try:
            cur.execute(sql)
            cols = [d[0].lower() for d in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
            return pd.DataFrame(rows, columns=cols)
        finally:
            try:
                cur.close()
            except Exception:
                pass
