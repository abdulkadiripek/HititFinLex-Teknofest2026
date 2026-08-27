from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BACKEND_DB = Path(__file__).resolve().parents[1] / "db"
if str(BACKEND_DB) not in sys.path:
    sys.path.insert(0, str(BACKEND_DB))

import provision


class StubResult:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class StubConnection:
    def __init__(self, handler=None):
        self.handler = handler or (lambda _query, _parameters: ())
        self.calls = []
        self.rollbacks = 0

    def execute(self, query, parameters=None):
        rendered = str(query)
        self.calls.append((rendered, parameters))
        rows = self.handler(rendered, parameters)
        return StubResult(rows)

    def rollback(self):
        self.rollbacks += 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def settings() -> provision.Settings:
    return provision.Settings(
        host="127.0.0.1",
        port=5432,
        database="hititfinlex_test",
        admin_user="postgres",
        admin_password="admin-password-long",
        migrator_user="hititfinlex_migrator",
        migrator_password="migrator-password-long",
        app_user="hititfinlex_app",
        app_password="runtime-password-long",
    )


def find_call(connection: StubConnection, fragment: str) -> int:
    for index, (query, _parameters) in enumerate(connection.calls):
        if fragment in query:
            return index
    raise AssertionError(f"SQL fragment was not executed: {fragment}")


class ProvisionSqlTest(unittest.TestCase):
    def test_existing_login_role_clears_every_elevated_flag(self):
        connection = StubConnection(
            lambda query, _parameters: [(1,)] if "SELECT 1 FROM pg_roles" in query else ()
        )

        provision.ensure_login_role(
            connection,
            "hititfinlex_app",
            "runtime-password-long",
        )

        statement = connection.calls[1][0]
        for fragment in (
            "NOSUPERUSER",
            "NOCREATEDB",
            "NOCREATEROLE",
            "NOINHERIT",
            "NOREPLICATION",
            "NOBYPASSRLS",
        ):
            self.assertIn(fragment, statement)

    def test_memberships_are_revoked_and_rechecked(self):
        membership_queries = 0

        def handler(query, _parameters):
            nonlocal membership_queries
            if "FROM pg_auth_members" in query:
                membership_queries += 1
                return [("legacy_admin",)] if membership_queries == 1 else ()
            return ()

        connection = StubConnection(handler)
        provision.revoke_role_memberships(connection, "hititfinlex_app")

        self.assertEqual(membership_queries, 2)
        revoke = connection.calls[1][0]
        self.assertIn("REVOKE ", revoke)
        self.assertIn("legacy_admin", revoke)
        self.assertIn("hititfinlex_app", revoke)
        self.assertIn("CASCADE", revoke)

    def test_bootstrap_revokes_legacy_database_and_schema_access(self):
        def maintenance_handler(query, _parameters):
            if "SELECT 1 FROM pg_roles" in query:
                return [(1,)]
            if "FROM pg_auth_members" in query:
                return ()
            if "SELECT 1 FROM pg_database" in query:
                return [(1,)]
            return ()

        maintenance = StubConnection(maintenance_handler)
        target = StubConnection()
        with (
            patch(
                "provision.psycopg.connect",
                side_effect=(maintenance, target),
            ),
            patch("provision.user_schema_names", return_value=["public"]),
            patch("provision.transfer_legacy_public_objects"),
        ):
            provision.bootstrap(settings())

        public_revoke = find_call(
            maintenance,
            "REVOKE ALL ON DATABASE ",
        )
        app_revoke = next(
            index
            for index, (query, _parameters) in enumerate(maintenance.calls)
            if "REVOKE ALL ON DATABASE " in query
            and "hititfinlex_app" in query
        )
        connect_grant = find_call(maintenance, "GRANT CONNECT ON DATABASE")
        self.assertLess(public_revoke, app_revoke)
        self.assertLess(app_revoke, connect_grant)
        self.assertGreaterEqual(find_call(maintenance, "SET search_path"), 0)
        self.assertGreaterEqual(find_call(target, "REASSIGN OWNED BY"), 0)
        schema_app_revoke = next(
            query
            for query, _parameters in target.calls
            if "REVOKE ALL ON SCHEMA " in query
            and "hititfinlex_app" in query
        )
        self.assertIn("public", schema_app_revoke)
        self.assertGreaterEqual(find_call(target, "GRANT USAGE ON SCHEMA public"), 0)

    def test_runtime_grants_revoke_before_exact_grants(self):
        connection = StubConnection()
        with (
            patch("provision.psycopg.connect", return_value=connection),
            patch("provision.user_schema_names", return_value=["public"]),
        ):
            provision.grant_runtime_access(settings())

        table_revoke = find_call(
            connection,
            "REVOKE ALL ON ALL TABLES IN SCHEMA ",
        )
        table_grant = find_call(
            connection,
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES",
        )
        sequence_revoke = find_call(
            connection,
            "REVOKE ALL ON ALL SEQUENCES IN SCHEMA ",
        )
        sequence_grant = find_call(
            connection,
            "GRANT USAGE, SELECT ON ALL SEQUENCES",
        )
        default_table_revoke = next(
            index
            for index, (query, _parameters) in enumerate(connection.calls)
            if "ALTER DEFAULT PRIVILEGES" in query
            and "REVOKE ALL ON " in query
            and "TABLES" in query
        )
        default_table_grant = next(
            index
            for index, (query, _parameters) in enumerate(connection.calls)
            if "ALTER DEFAULT PRIVILEGES" in query
            and "GRANT SELECT, INSERT, UPDATE, DELETE" in query
        )
        self.assertLess(table_revoke, table_grant)
        self.assertLess(sequence_revoke, sequence_grant)
        self.assertLess(default_table_revoke, default_table_grant)
        self.assertIn("CASCADE", connection.calls[table_revoke][0])
        self.assertIn("CASCADE", connection.calls[sequence_revoke][0])
        self.assertGreaterEqual(
            find_call(
                connection,
                "REVOKE ALL ON TABLE hititfinlex_schema_migrations",
            ),
            0,
        )


class VerifyRuntimeRoleTest(unittest.TestCase):
    def verifier_connection(self, override=None):
        table_names = [
            *provision.migrate.REQUIRED_TABLES,
            "hititfinlex_schema_migrations",
        ]
        relation_by_oid = {
            index + 1000: name for index, name in enumerate(table_names)
        }

        def handler(query, parameters):
            if override is not None:
                overridden = override(query, parameters, relation_by_oid)
                if overridden is not None:
                    return overridden
            if "SELECT" in query and "rolsuper" in query:
                return [(False, False, False, False, False, False, True)]
            if "FROM pg_auth_members" in query:
                return ()
            if "has_database_privilege" in query:
                return [(True, False, False)]
            if "FROM pg_database" in query:
                return [("hititfinlex_migrator",)]
            if "WHERE nspname = 'public'" in query:
                return [("hititfinlex_migrator",)]
            if "relation.relowner <>" in query:
                return ()
            if "WITH runtime_role AS" in query:
                return [(0, 0, 0, 0)]
            if "SELECT nspname" in query and "FROM pg_namespace" in query:
                return [("public",)]
            if "has_schema_privilege" in query:
                return [(True, False)]
            if "relation.relkind IN ('r', 'p', 'v', 'm', 'f')" in query:
                return [
                    ("public", name, relation_oid)
                    for relation_oid, name in relation_by_oid.items()
                ]
            if "has_table_privilege" in query:
                relation_name = relation_by_oid[int(parameters[0])]
                if relation_name == "hititfinlex_schema_migrations":
                    return [(False,) * 8]
                return [(True, True, True, True, False, False, False, False)]
            if "relation.relkind = 'S'" in query:
                return [("public", "documents_id_seq", 9001)]
            if "has_sequence_privilege" in query:
                return [(True, True, False)]
            if "CREATE TABLE public.hititfinlex_permission_probe" in query:
                raise provision.errors.InsufficientPrivilege("denied")
            return ()

        return StubConnection(handler)

    def run_verifier(self, connection):
        with patch("provision.psycopg.connect", return_value=connection):
            provision.verify_runtime_role(settings())

    def test_verifier_accepts_only_exact_least_privilege_role(self):
        connection = self.verifier_connection()
        self.run_verifier(connection)
        self.assertEqual(connection.rollbacks, 1)

    def test_verifier_rejects_bypassrls_and_membership(self):
        def bypass_override(query, _parameters, _relations):
            if "rolsuper" in query:
                return [(False, False, False, False, True, False, True)]
            return None

        with self.assertRaises(RuntimeError):
            self.run_verifier(self.verifier_connection(bypass_override))

        def membership_override(query, _parameters, _relations):
            if "FROM pg_auth_members" in query:
                return [("legacy_admin",)]
            return None

        with self.assertRaises(RuntimeError):
            self.run_verifier(self.verifier_connection(membership_override))

    def test_verifier_rejects_database_create_or_temp(self):
        def override(query, _parameters, _relations):
            if "has_database_privilege" in query:
                return [(True, True, True)]
            return None

        with self.assertRaises(RuntimeError):
            self.run_verifier(self.verifier_connection(override))

    def test_verifier_checks_each_table_privilege_separately(self):
        def override(query, parameters, relations):
            if "has_table_privilege" in query:
                relation_name = relations[int(parameters[0])]
                if relation_name != "hititfinlex_schema_migrations":
                    return [
                        (True, True, True, False, False, False, False, False)
                    ]
            return None

        with self.assertRaises(RuntimeError):
            self.run_verifier(self.verifier_connection(override))

    def test_verifier_rejects_runtime_object_ownership(self):
        def override(query, _parameters, _relations):
            if "WITH runtime_role AS" in query:
                return [(1, 0, 0, 0)]
            return None

        with self.assertRaises(RuntimeError):
            self.run_verifier(self.verifier_connection(override))


if __name__ == "__main__":
    unittest.main()
