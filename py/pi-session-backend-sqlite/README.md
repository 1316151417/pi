# pi-session-backend-sqlite

Native Python port of `packages/session-backends/sqlite-node` (0.85.1).
The source package's `node` suffix identifies its runtime adapter; this package
uses Python's standard-library `sqlite3`, without starting Node or translating
SQL through a service.

## Public interface

`SqliteSessionRepo(SqliteSessionRepoOptions(directory=..., database_factory=create_sqlite_factory()))`
provides `create`, `open`, `list`, `delete`, `fork`, and `close`. Supply the shared
`pi_chord.context.Context` argument required by the Session contracts. Operations
return reusable awaitables and start when called within an active asyncio loop.
`database_path` selects a shared container instead of the default file per session.

The package also exports `SqliteStorage`, its options and snapshot types,
`SqliteOpenSession`, metadata, database/statement/factory protocols, and
`SqliteRunResult`. `create_node_sqlite_factory` and `wrap_node_sqlite_database`
retain the original export names in snake_case; they use the same native Python
implementation as `create_sqlite_factory` and `wrap_sqlite_database`.

`SqlQuery(query_text, params)` supports `exec`, `run`, `get`, `all`, and `iterate`.
Use `sql(("SELECT * FROM sessions WHERE id = ", ""), session_id)` for the source's
tagged-template operation. Nested `SqlQuery` instances are trusted fragments;
other interpolations become bound parameters. `join_sql_fragments` retains
parameter order. A connection passed to `wrap_sqlite_database` must use
autocommit (`isolation_level=None`), because this adapter owns transaction boundaries.

## Stored behavior

The schema and triggers are copied verbatim. Transactions use `BEGIN IMMEDIATE`,
reject awaitable callbacks, and roll back on failure. Entries and usage share an
ID namespace, parents must already exist at insertion, and a failed commit leaves
the sequence and projections unchanged. Branch indexes retain compaction sharing,
inclusive stop boundaries, exclusive cursors, ordering, and per-segment limits.

Safe session IDs use `<id>.sqlite`; other IDs use `~` followed by unpadded
base64url of the original UTF-16LE code units. Returned paths are canonical.
Repository open/delete reject foreign paths and never create missing databases.
Discovery opens read-only and skips corrupt, unrelated, and incompatible files.
Shared-container deletion removes only that session's rows.

Local ownership excludes overlapping create/open/fork/delete operations for an
ID. Local source forks join the source commit queue. Foreign sources use their
exact path and one read-only deferred transaction for a consistent snapshot.
Forks copy the selected entries and projected scalar values, rebuild branch
indexes and message counts, and start with an empty usage ledger and zero usage.
As in the source, this is not a cross-process writer lease.

Session close waits for admitted operations and explicitly held mutations before
closing storage and the database. Storage close waits for queued commits without
closing the connection itself. Repeated close calls return the same awaitable.
Repository close settles all open-session closes before propagating one failure
or grouping multiple failures. Pending repository creates are not added to the
source's open-session close snapshot; this source behavior is retained.

## Source mapping

| TypeScript source relative to `src/` | Python module |
| --- | --- |
| `index.ts` | `adapter.py`, `__init__.py` |
| `sqlite/index.ts` | `__init__.py` |
| `sqlite/types.ts` | `types.py` |
| `sqlite/sql.ts` | `sql.py` |
| `sqlite/migrations.ts` | `migrations.py` |
| `sqlite/migrations/001_initial.sql` | `migrations/001_initial.sql` (unchanged) |
| `sqlite/session/{entries,usage-ledger,session-row,session-sequences,session-stats}.ts` | `_rows.py` |
| `sqlite/session/values.ts` | `_values.py` |
| `sqlite/session/branch-entries.ts` | `_branches.py` |
| `sqlite/storage.ts` | `storage.py` |
| `sqlite/session.ts` | `session.py` |
| `sqlite/repo.ts` | `repo.py` |

`_json.py` and `_async.py` adapt JavaScript object and Promise semantics.
The Node binding behavior was additionally compared with official Node v22.19.0
`src/node_sqlite.cc` for binding, row conversion, preparation, and execution.

## Runtime adaptations and unverified boundaries

- JSON is parsed without model reconstruction, default insertion, integer
  truncation, or removal of unknown fields. Returned dictionaries retain wire
  keys and additionally support snake_case attribute access; nested records use
  the same view. They are structurally compatible records, not concrete
  `MessageEntry`, `AssistantMessage`, or `Usage` dataclass instances. Most harness
  readers use structural access. The legacy top-level agent has an
  `isinstance(AssistantMessage)` check on stream event messages; applications that
  feed stored records into that separate event path must account for this boundary.
- For optional JSON payloads, `pi_ai.types.UNDEFINED` means omitted and `None`
  means explicit JSON null. Existing upstream Entry/UsageRow dataclasses still
  default some `data`/`details` fields to `None`; the backend cannot infer whether
  their caller omitted them. It writes those defaults as explicit null. Optional
  typed `terminate`/`usage` dataclass fields retain the existing None-as-absent
  convention. Raw wire dictionaries preserve missing keys and explicit null.
- Parsed JSON numbers use JavaScript binary64 semantics. SQLite parameters accept
  numbers, strings, null and byte buffers. Python `int` and `float` bind as JS
  Number; booleans are rejected. A distinct JS BigInt input type is not exposed.
  SQLite integer results outside the JS safe-integer range raise `OverflowError`.
  Blob results are `bytes`; absent rows and optional Python API results use `None`.
- Python does not expose SQLite prepare independently of execution. `prepare`
  validates with `EXPLAIN` and NULL bindings, then executes the actual statement
  on its first operation. This retains early compilation errors without running
  the actual statement during preparation. The underlying prepared-statement
  lifetime and overlapping iterators follow Python cursor facilities rather than
  Node's native StatementSync object. Empty/comment-only prepared SQL and exotic
  SQLite parameter names outside the tokenizer remain unverified adapter edges.
- The linked SQLite version, platform filesystem behavior, native exception
  classes and SQLite error strings may differ from Node's bundled SQLite.
  Multiple close failures use `BaseExceptionGroup`/`ExceptionGroup` instead of
  JavaScript `AggregateError`. All SQLite calls stay on the calling event-loop
  thread; filesystem work runs through `asyncio.to_thread`. Waiter cancellation
  does not cancel shared admitted operations or the commit queue. Exact microtask
  turn counts differ between JavaScript and asyncio.
- Dependencies are the existing pi-agent-core, pi-ai, and pi-chord packages;
  no external SQLite dependency is required. Their pre-existing model, commit,
  session and fork behavior is reused rather than independently reimplemented.

Implementation only: no tests, import/compile checks, SQL probes, or database
creation were run during this port. Unified testing is deferred as requested.
