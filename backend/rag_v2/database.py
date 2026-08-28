from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .settings import RagV2Settings


class RagDatabasePool:
    def __init__(self, settings: RagV2Settings) -> None:
        self._settings = settings
        self._pool = ConnectionPool(
            conninfo="",
            kwargs={
                "host": settings.db_host,
                "port": settings.db_port,
                "dbname": settings.db_name,
                "user": settings.db_user,
                "password": settings.db_password,
                "row_factory": dict_row,
                "connect_timeout": max(
                    1, int(settings.db_pool_timeout_seconds)
                ),
            },
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            timeout=settings.db_pool_timeout_seconds,
            open=False,
            name="hititfinlex-rag-v2",
        )

    def open(self) -> None:
        self._pool.open(wait=True, timeout=self._settings.db_pool_timeout_seconds)

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self._pool.connection() as connection:
            yield connection

    def check(self) -> None:
        self._pool.check()

    def require_relations(self, relation_names: tuple[str, ...]) -> None:
        with self.connection() as connection:
            missing = connection.execute(
                """
                SELECT requested.name
                FROM UNNEST(%s::TEXT[]) AS requested(name)
                WHERE TO_REGCLASS('public.' || requested.name) IS NULL
                ORDER BY requested.name
                """,
                (list(relation_names),),
            ).fetchall()
        if missing:
            raise RuntimeError("RAG V2 database schema is not ready")

    def stats(self) -> dict[str, int]:
        return dict(self._pool.get_stats())
