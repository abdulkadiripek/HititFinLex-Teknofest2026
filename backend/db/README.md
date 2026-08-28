# PostgreSQL 18 migration and provisioning

The application runs without Docker. Docker is optional and is not required
by these commands. Use three distinct roles:

- bootstrap administrator: role/database creation and trusted extension setup;
- migrator: owns the database, `public` schema, and application objects;
- runtime application role: CRUD and sequence use only, with no schema DDL.

`db/provision.py all` creates or hardens the roles, installs `vector` and
`unaccent` in `public`, transfers legacy application objects to the migrator,
applies checksum-verified migrations, grants the exact runtime privileges, and
verifies the resulting least-privilege boundary.

The migrator uses `public, pg_catalog` so unqualified migration DDL is created
in its owned application schema. The runtime role uses `pg_catalog, public`;
because it has no `CREATE` privilege in `public`, untrusted runtime objects
cannot shadow catalog functions.

## Required environment

Keep real values only in the ignored root `.env` or a secret manager. Never
put credentials in command output, documentation, fixtures, or source files.

```env
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=katilim_finans
DB_USER=hititfinlex_app
DB_PASSWORD=<runtime-secret>

POSTGRES_ADMIN_USER=postgres
POSTGRES_ADMIN_PASSWORD=<bootstrap-secret>
DB_MIGRATOR_USER=hititfinlex_migrator
DB_MIGRATOR_PASSWORD=<migrator-secret>
```

`POSTGRES_DB` is accepted as an alias for the existing `DB_NAME` setting. If
both are set, they must be identical. Admin, migrator, and runtime roles must
be distinct, and migrator/runtime passwords must differ.

## Provision without Docker

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe db\provision.py all
```

Individual operations are also available:

```powershell
.\.venv\Scripts\python.exe db\provision.py bootstrap
.\.venv\Scripts\python.exe db\provision.py grants
.\.venv\Scripts\python.exe db\provision.py verify
```

`bootstrap` and `all` require the bootstrap administrator variables. The API
must continue to run with `DB_USER`; do not replace it with an administrator or
migrator account.

## Apply migrations with an existing migrator

When roles already exist, set `DATABASE_URL` temporarily to the migrator DSN.
Do not use the runtime application DSN for `up` or `smoke`:

```powershell
$env:DATABASE_URL = "postgresql://<migrator>:<secret>@127.0.0.1:5432/katilim_finans"
.\.venv\Scripts\python.exe db\migrate.py check
.\.venv\Scripts\python.exe db\migrate.py status
.\.venv\Scripts\python.exe db\migrate.py up
.\.venv\Scripts\python.exe db\migrate.py smoke
Remove-Item Env:DATABASE_URL
```

`check` is offline. `up` records file/version/SHA-256 triples and refuses a
changed migration that was already applied. The RAG V2 SQL surfaces use
idempotent table, index, extension, function, and trigger declarations.
`smoke` verifies
PostgreSQL major version 18, pgvector, `unaccent`, `vector(1024)`, the RAG V2
classification/state columns, GIN/B-tree indexes, lexical trigger, and session
security columns.

Migration `0003_rag_v2.sql` adds:

- `rag_chunks`, with weighted `simple + unaccent` `tsvector`, GIN indexes,
  hard-filter columns, classification confidence/status, product arrays, and
  structured facts;
- `rag_sessions`, with random-token hashes, optional owner hashes, TTL, and
  revocation time;
- `rag_messages`, `rag_session_state`, and `rag_turn_evidence`, with composite
  session/time indexes.

Migration `0004_rag_v2_conversation.sql` extends the `rag_messages.status`
constraint with the non-evidentiary `conversational` status used by EVREN
casual-chat turns. It does not weaken citation validation for financial turns.

## Tests

Static checks do not need a database:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_migrations tests.test_provision
```

For the transactional PostgreSQL idempotence test, point only to a dedicated
PostgreSQL 18 test database whose baseline migrations are present:

```powershell
$env:RAG_V2_TEST_DATABASE_URL = "postgresql://<migrator>:<secret>@127.0.0.1:5432/hititfinlex_test"
.\.venv\Scripts\python.exe -m unittest tests.test_migrations.RagV2PostgresMigrationTest
Remove-Item Env:RAG_V2_TEST_DATABASE_URL
```

The integration test executes `0003_rag_v2.sql` twice in one transaction and
always rolls the transaction back. It is skipped when the dedicated test DSN
is absent.
