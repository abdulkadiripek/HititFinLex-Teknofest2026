"""Create separated PostgreSQL migration/runtime roles and apply migrations."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

import migrate


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = BACKEND_DIR.parent
load_dotenv(REPOSITORY_DIR / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)

SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
PLACEHOLDER_MARKERS = ("change_me", "changeme", "replace_me")


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def identifier(name: str) -> str:
    value = required(name)
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise RuntimeError(f"Unsafe PostgreSQL identifier in {name}: {value!r}")
    return value


def aliased_identifier(*names: str) -> str:
    supplied = {
        name: os.getenv(name, "").strip()
        for name in names
        if os.getenv(name, "").strip()
    }
    if not supplied:
        raise RuntimeError(
            "Missing required environment variable; set one of: "
            + ", ".join(names)
        )
    invalid = [
        name for name, value in supplied.items()
        if not SAFE_IDENTIFIER.fullmatch(value)
    ]
    if invalid:
        raise RuntimeError(
            "Unsafe PostgreSQL identifier in: " + ", ".join(invalid)
        )
    values = set(supplied.values())
    if len(values) != 1:
        raise RuntimeError(
            "Aliased PostgreSQL identifiers must match: "
            + ", ".join(names)
        )
    return next(iter(values))


def password(name: str) -> str:
    value = required(name)
    normalized = value.casefold()
    if len(value) < 16 or normalized.startswith(PLACEHOLDER_MARKERS):
        raise RuntimeError(
            f"{name} must be a non-placeholder secret of at least 16 characters"
        )
    return value


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    database: str
    admin_user: str
    admin_password: str
    migrator_user: str
    migrator_password: str
    app_user: str
    app_password: str

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            host=os.getenv("DB_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("DB_PORT", "5432")),
            database=aliased_identifier("POSTGRES_DB", "DB_NAME"),
            admin_user=identifier("POSTGRES_ADMIN_USER"),
            admin_password=password("POSTGRES_ADMIN_PASSWORD"),
            migrator_user=identifier("DB_MIGRATOR_USER"),
            migrator_password=password("DB_MIGRATOR_PASSWORD"),
            app_user=identifier("DB_USER"),
            app_password=password("DB_PASSWORD"),
        )
        if len({settings.admin_user, settings.migrator_user, settings.app_user}) != 3:
            raise RuntimeError("Admin, migrator, and app roles must be distinct")
        if settings.migrator_password == settings.app_password:
            raise RuntimeError("Migrator and app roles must use distinct passwords")
        return settings

    def conninfo(self, *, user: str, secret: str, database: str | None = None) -> str:
        return make_conninfo(
            host=self.host,
            port=self.port,
            dbname=database or self.database,
            user=user,
            password=secret,
            connect_timeout=10,
        )


def ensure_login_role(connection, role: str, secret: str) -> None:
    exists = connection.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)
    ).fetchone()
    statement = "ALTER ROLE" if exists else "CREATE ROLE"
    connection.execute(
        sql.SQL(
            f"{statement} {{}} LOGIN PASSWORD {{}} "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT "
            "NOREPLICATION NOBYPASSRLS"
        ).format(sql.Identifier(role), sql.Literal(secret))
    )


def revoke_role_memberships(connection, role: str) -> None:
    """Remove every direct membership that could allow SET ROLE escalation."""
    memberships = connection.execute(
        """
        SELECT granted_role.rolname
        FROM pg_auth_members AS membership
        JOIN pg_roles AS granted_role
          ON granted_role.oid = membership.roleid
        JOIN pg_roles AS member_role
          ON member_role.oid = membership.member
        WHERE member_role.rolname = %s
        """,
        (role,),
    ).fetchall()
    for (granted_role,) in memberships:
        connection.execute(
            sql.SQL("REVOKE {} FROM {} CASCADE").format(
                sql.Identifier(granted_role),
                sql.Identifier(role),
            )
        )
    remaining = connection.execute(
        """
        SELECT granted_role.rolname
        FROM pg_auth_members AS membership
        JOIN pg_roles AS granted_role
          ON granted_role.oid = membership.roleid
        JOIN pg_roles AS member_role
          ON member_role.oid = membership.member
        WHERE member_role.rolname = %s
        """,
        (role,),
    ).fetchall()
    if remaining:
        raise RuntimeError(
            f"Runtime role retains role memberships: {remaining}"
        )


def user_schema_names(connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT nspname
        FROM pg_namespace
        WHERE nspname !~ '^pg_'
          AND nspname <> 'information_schema'
        ORDER BY nspname
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def transfer_legacy_public_objects(connection, owner: str) -> None:
    """Give the migrator ownership of legacy app objects, not extension internals."""
    rows = connection.execute(
        """
        SELECT relation.relkind, relation.relname
        FROM pg_class AS relation
        JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relkind IN ('r', 'p', 'S', 'v', 'm')
          AND NOT EXISTS (
              SELECT 1
              FROM pg_depend AS dependency
              WHERE dependency.classid = 'pg_class'::regclass
                AND dependency.objid = relation.oid
                AND dependency.deptype = 'e'
          )
          AND NOT (
              relation.relkind = 'S'
              AND EXISTS (
                  SELECT 1
                  FROM pg_depend AS ownership
                  WHERE ownership.classid = 'pg_class'::regclass
                    AND ownership.objid = relation.oid
                    AND ownership.refclassid = 'pg_class'::regclass
                    AND ownership.deptype IN ('a', 'i')
              )
          )
        ORDER BY
            CASE WHEN relation.relkind IN ('r', 'p') THEN 0 ELSE 1 END,
            relation.relkind,
            relation.relname
        """
    ).fetchall()
    kinds = {
        "r": "TABLE",
        "p": "TABLE",
        "S": "SEQUENCE",
        "v": "VIEW",
        "m": "MATERIALIZED VIEW",
    }
    for relation_kind, relation_name in rows:
        connection.execute(
            sql.SQL("ALTER {} {} OWNER TO {}").format(
                sql.SQL(kinds[relation_kind]),
                sql.Identifier("public", relation_name),
                sql.Identifier(owner),
            )
        )


def bootstrap(settings: Settings) -> None:
    admin_maintenance = settings.conninfo(
        user=settings.admin_user,
        secret=settings.admin_password,
        database="postgres",
    )
    with psycopg.connect(admin_maintenance, autocommit=True) as connection:
        ensure_login_role(
            connection,
            settings.migrator_user,
            settings.migrator_password,
        )
        ensure_login_role(connection, settings.app_user, settings.app_password)
        revoke_role_memberships(connection, settings.app_user)
        connection.execute(
            sql.SQL("ALTER ROLE {} CONNECTION LIMIT 20").format(
                sql.Identifier(settings.app_user)
            )
        )
        database_exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.database,),
        ).fetchone()
        if not database_exists:
            connection.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(settings.database),
                    sql.Identifier(settings.migrator_user),
                )
            )
        else:
            connection.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(settings.database),
                    sql.Identifier(settings.migrator_user),
                )
            )
        connection.execute(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC CASCADE").format(
                sql.Identifier(settings.database)
            )
        )
        connection.execute(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM {} CASCADE").format(
                sql.Identifier(settings.database),
                sql.Identifier(settings.app_user),
            )
        )
        connection.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(
                sql.Identifier(settings.database),
                sql.Identifier(settings.migrator_user),
                sql.Identifier(settings.app_user),
            )
        )
        connection.execute(
            sql.SQL(
                "ALTER ROLE {} IN DATABASE {} "
                "SET search_path TO public, pg_catalog"
            ).format(
                sql.Identifier(settings.migrator_user),
                sql.Identifier(settings.database),
            )
        )
        connection.execute(
            sql.SQL(
                "ALTER ROLE {} IN DATABASE {} "
                "SET search_path TO pg_catalog, public"
            ).format(
                sql.Identifier(settings.app_user),
                sql.Identifier(settings.database),
            )
        )

    admin_target = settings.conninfo(
        user=settings.admin_user,
        secret=settings.admin_password,
    )
    with psycopg.connect(admin_target, autocommit=True) as connection:
        connection.execute(
            "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"
        )
        connection.execute(
            "CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public"
        )
        connection.execute(
            sql.SQL("REASSIGN OWNED BY {} TO {}").format(
                sql.Identifier(settings.app_user),
                sql.Identifier(settings.migrator_user),
            )
        )
        transfer_legacy_public_objects(connection, settings.migrator_user)
        connection.execute(
            sql.SQL("ALTER SCHEMA public OWNER TO {}").format(
                sql.Identifier(settings.migrator_user)
            )
        )
        for schema_name in user_schema_names(connection):
            connection.execute(
                sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )
            connection.execute(
                sql.SQL("REVOKE ALL ON SCHEMA {} FROM {} CASCADE").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(settings.app_user),
                )
            )
        connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(
                sql.Identifier(settings.app_user)
            )
        )


def migrate_schema(settings: Settings) -> None:
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = settings.conninfo(
        user=settings.migrator_user,
        secret=settings.migrator_password,
    )
    try:
        migrate.migrate_up(migrate.load_manifest())
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def grant_runtime_access(settings: Settings) -> None:
    # Use the bootstrap administrator for grants so this also upgrades legacy
    # databases whose existing tables are still owned by the old postgres
    # role. Newly migrated objects remain owned by the dedicated migrator.
    admin_target = settings.conninfo(
        user=settings.admin_user,
        secret=settings.admin_password,
    )
    with psycopg.connect(admin_target, autocommit=True) as connection:
        app_role = sql.Identifier(settings.app_user)
        public_role = sql.SQL("PUBLIC")
        for schema_name in user_schema_names(connection):
            schema = sql.Identifier(schema_name)
            for grantee in (public_role, app_role):
                connection.execute(
                    sql.SQL(
                        "REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {} CASCADE"
                    ).format(schema, grantee)
                )
                connection.execute(
                    sql.SQL(
                        "REVOKE ALL ON ALL SEQUENCES IN SCHEMA {} FROM {} CASCADE"
                    ).format(schema, grantee)
                )

        migrator_role = sql.Identifier(settings.migrator_user)
        for object_kind in ("TABLES", "SEQUENCES"):
            connection.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                    "REVOKE ALL ON {} FROM PUBLIC CASCADE"
                ).format(migrator_role, sql.SQL(object_kind))
            )
            connection.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                    "REVOKE ALL ON {} FROM {} CASCADE"
                ).format(migrator_role, sql.SQL(object_kind), app_role)
            )

        connection.execute(
            sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                "IN SCHEMA public TO {}"
            ).format(app_role)
        )
        connection.execute(
            sql.SQL(
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}"
            ).format(app_role)
        )
        connection.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}"
            ).format(migrator_role, app_role)
        )
        connection.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
                "GRANT USAGE, SELECT ON SEQUENCES TO {}"
            ).format(migrator_role, app_role)
        )
        connection.execute(
            sql.SQL(
                "REVOKE ALL ON TABLE hititfinlex_schema_migrations "
                "FROM {} CASCADE"
            ).format(app_role)
        )


def verify_runtime_role(settings: Settings) -> None:
    app_target = settings.conninfo(
        user=settings.app_user,
        secret=settings.app_password,
    )
    with psycopg.connect(app_target) as connection:
        role_flags = connection.execute(
            """
            SELECT
                rolsuper,
                rolcreatedb,
                rolcreaterole,
                rolreplication,
                rolbypassrls,
                rolinherit,
                rolcanlogin
            FROM pg_roles
            WHERE rolname = current_user
            """
        ).fetchone()
        expected_role_flags = (False, False, False, False, False, False, True)
        if role_flags != expected_role_flags:
            raise RuntimeError(
                f"Runtime role has unexpected role flags: {role_flags}"
            )

        memberships = connection.execute(
            """
            SELECT granted_role.rolname
            FROM pg_auth_members AS membership
            JOIN pg_roles AS granted_role
              ON granted_role.oid = membership.roleid
            WHERE membership.member = (
                SELECT oid FROM pg_roles WHERE rolname = current_user
            )
            ORDER BY granted_role.rolname
            """
        ).fetchall()
        if memberships:
            raise RuntimeError(
                f"Runtime role retains role memberships: {memberships}"
            )

        database_privileges = connection.execute(
            """
            SELECT
                has_database_privilege(current_user, current_database(), 'CONNECT'),
                has_database_privilege(current_user, current_database(), 'CREATE'),
                has_database_privilege(current_user, current_database(), 'TEMP')
            """
        ).fetchone()
        if database_privileges != (True, False, False):
            raise RuntimeError(
                "Runtime role has unexpected database privileges: "
                f"{database_privileges}"
            )

        database_owner = connection.execute(
            """
            SELECT pg_get_userbyid(datdba)
            FROM pg_database
            WHERE datname = current_database()
            """
        ).fetchone()
        if database_owner != (settings.migrator_user,):
            raise RuntimeError(
                f"Unexpected database owner: {database_owner}"
            )

        public_schema_owner = connection.execute(
            """
            SELECT pg_get_userbyid(nspowner)
            FROM pg_namespace
            WHERE nspname = 'public'
            """
        ).fetchone()
        if public_schema_owner != (settings.migrator_user,):
            raise RuntimeError(
                f"Unexpected public schema owner: {public_schema_owner}"
            )

        unexpected_public_owners = connection.execute(
            """
            SELECT relation.relname, pg_get_userbyid(relation.relowner)
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relkind IN ('r', 'p', 'S', 'v', 'm', 'f')
              AND relation.relowner <> (
                  SELECT oid FROM pg_roles WHERE rolname = %s
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_depend AS dependency
                  WHERE dependency.classid = 'pg_class'::regclass
                    AND dependency.objid = relation.oid
                    AND dependency.deptype = 'e'
              )
            ORDER BY relation.relname
            """,
            (settings.migrator_user,),
        ).fetchall()
        if unexpected_public_owners:
            raise RuntimeError(
                "Public relations retain unexpected owners: "
                f"{unexpected_public_owners}"
            )

        app_owned_objects = connection.execute(
            """
            WITH runtime_role AS (
                SELECT oid FROM pg_roles WHERE rolname = current_user
            )
            SELECT
                (SELECT COUNT(*) FROM pg_class, runtime_role
                 WHERE relowner = runtime_role.oid),
                (SELECT COUNT(*) FROM pg_proc, runtime_role
                 WHERE proowner = runtime_role.oid),
                (SELECT COUNT(*) FROM pg_type, runtime_role
                 WHERE typowner = runtime_role.oid),
                (SELECT COUNT(*) FROM pg_namespace, runtime_role
                 WHERE nspowner = runtime_role.oid)
            """
        ).fetchone()
        if app_owned_objects != (0, 0, 0, 0):
            raise RuntimeError(
                "Runtime role still owns database objects "
                f"(relations, functions, types, schemas): {app_owned_objects}"
            )

        schemas = user_schema_names(connection)
        if "public" not in schemas:
            raise RuntimeError("Required schema public was not found")
        for schema_name in schemas:
            schema_privileges = connection.execute(
                """
                SELECT
                    has_schema_privilege(current_user, %s, 'USAGE'),
                    has_schema_privilege(current_user, %s, 'CREATE')
                """,
                (schema_name, schema_name),
            ).fetchone()
            expected_schema_privileges = (
                (True, False)
                if schema_name == "public"
                else (False, False)
            )
            if schema_privileges != expected_schema_privileges:
                raise RuntimeError(
                    f"Runtime role has unexpected privileges on schema "
                    f"{schema_name}: {schema_privileges}"
                )

        relations = connection.execute(
            """
            SELECT namespace.nspname, relation.relname, relation.oid
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = ANY(%s)
              AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_depend AS dependency
                  WHERE dependency.classid = 'pg_class'::regclass
                    AND dependency.objid = relation.oid
                    AND dependency.deptype = 'e'
              )
            ORDER BY namespace.nspname, relation.relname
            """,
            (schemas,),
        ).fetchall()
        public_relation_names = {
            str(relation_name)
            for schema_name, relation_name, _relation_oid in relations
            if schema_name == "public"
        }
        missing_tables = sorted(
            set(migrate.REQUIRED_TABLES) - public_relation_names
        )
        if missing_tables:
            raise RuntimeError(
                "Required runtime tables are missing: "
                + ", ".join(missing_tables)
            )

        for schema_name, relation_name, relation_oid in relations:
            table_privileges = connection.execute(
                """
                SELECT
                    has_table_privilege(current_user, %s, 'SELECT'),
                    has_table_privilege(current_user, %s, 'INSERT'),
                    has_table_privilege(current_user, %s, 'UPDATE'),
                    has_table_privilege(current_user, %s, 'DELETE'),
                    has_table_privilege(current_user, %s, 'TRUNCATE'),
                    has_table_privilege(current_user, %s, 'REFERENCES'),
                    has_table_privilege(current_user, %s, 'TRIGGER'),
                    has_any_column_privilege(current_user, %s, 'REFERENCES')
                """,
                (relation_oid,) * 8,
            ).fetchone()
            grants_crud = (
                schema_name == "public"
                and relation_name != "hititfinlex_schema_migrations"
            )
            expected_table_privileges = (
                (True, True, True, True, False, False, False, False)
                if grants_crud
                else (False,) * 8
            )
            if table_privileges != expected_table_privileges:
                raise RuntimeError(
                    f"Runtime role has unexpected table privileges on "
                    f"{schema_name}.{relation_name}: {table_privileges}"
                )

        sequences = connection.execute(
            """
            SELECT namespace.nspname, relation.relname, relation.oid
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = ANY(%s)
              AND relation.relkind = 'S'
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_depend AS dependency
                  WHERE dependency.classid = 'pg_class'::regclass
                    AND dependency.objid = relation.oid
                    AND dependency.deptype = 'e'
              )
            ORDER BY namespace.nspname, relation.relname
            """,
            (schemas,),
        ).fetchall()
        for schema_name, sequence_name, sequence_oid in sequences:
            sequence_privileges = connection.execute(
                """
                SELECT
                    has_sequence_privilege(current_user, %s, 'USAGE'),
                    has_sequence_privilege(current_user, %s, 'SELECT'),
                    has_sequence_privilege(current_user, %s, 'UPDATE')
                """,
                (sequence_oid,) * 3,
            ).fetchone()
            expected_sequence_privileges = (
                (True, True, False)
                if schema_name == "public"
                else (False, False, False)
            )
            if sequence_privileges != expected_sequence_privileges:
                raise RuntimeError(
                    f"Runtime role has unexpected sequence privileges on "
                    f"{schema_name}.{sequence_name}: {sequence_privileges}"
                )

        try:
            connection.execute(
                "CREATE TABLE public.hititfinlex_permission_probe (id integer)"
            )
        except errors.InsufficientPrivilege:
            connection.rollback()
        else:
            connection.rollback()
            raise RuntimeError("Runtime role unexpectedly has schema DDL privilege")

    print(
        f"least-privilege role passed ({settings.app_user}: CRUD, no schema DDL)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bootstrap", "grants", "verify", "all"))
    args = parser.parse_args()
    settings = Settings.from_environment()

    if args.command in {"bootstrap", "all"}:
        bootstrap(settings)
    if args.command == "all":
        migrate_schema(settings)
        grant_runtime_access(settings)
        verify_runtime_role(settings)
    elif args.command == "grants":
        grant_runtime_access(settings)
    elif args.command == "verify":
        verify_runtime_role(settings)


def run_cli() -> int:
    try:
        main()
    except psycopg.Error:
        print(
            "database provisioning failed: database operation unavailable",
            file=sys.stderr,
        )
        return 2
    except (RuntimeError, ValueError) as error:
        print(f"database provisioning failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
