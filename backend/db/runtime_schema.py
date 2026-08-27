from __future__ import annotations

from collections.abc import Iterable


def require_migrated_tables(cursor, table_names: Iterable[str]) -> None:
    """Fail closed when required public tables were not created by migrations."""
    names = tuple(table_names)
    if not names:
        return
    placeholders = ", ".join("to_regclass(%s) IS NOT NULL" for _ in names)
    cursor.execute(
        f"SELECT {placeholders}",
        tuple(f"public.{table_name}" for table_name in names),
    )
    row = cursor.fetchone()
    missing = [
        table_name
        for index, table_name in enumerate(names)
        if row is None or index >= len(row) or not bool(row[index])
    ]
    if missing:
        qualified = ", ".join(f"public.{table_name}" for table_name in missing)
        raise RuntimeError(
            f"Required migrated table(s) were not found: {qualified}. "
            "Run the backend database migrations before this operation."
        )


def require_migrated_columns(
    cursor,
    table_name: str,
    column_names: Iterable[str],
) -> None:
    """Fail closed when a required migration column is unavailable."""
    names = tuple(column_names)
    if not names:
        return
    checks = ", ".join(
        "EXISTS ("
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' "
        "AND table_name = %s AND column_name = %s"
        ")"
        for _ in names
    )
    parameters = tuple(
        value
        for column_name in names
        for value in (table_name, column_name)
    )
    cursor.execute(f"SELECT {checks}", parameters)
    row = cursor.fetchone()
    missing = [
        column_name
        for index, column_name in enumerate(names)
        if row is None or index >= len(row) or not bool(row[index])
    ]
    if missing:
        qualified = ", ".join(
            f"public.{table_name}.{column_name}" for column_name in missing
        )
        raise RuntimeError(
            f"Required migrated column(s) were not found: {qualified}. "
            "Run the backend database migrations before this operation."
        )
