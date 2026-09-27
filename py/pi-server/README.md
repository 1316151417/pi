# pi-server (Python)

Python port of the core remote Session server in `packages/server/src`.
It retains the protocol v8 handshake, bounded RPC errors, cancellation,
Chord service snapshots and ordered updates, presentation-scoped Session
attachments, shutdown cleanup, and Unix-domain socket transport.

`Server` accepts any byte listener implementing `ServerListener`.
`pi_server.unix` exposes `create_unix_listener`, `create_unix_server`, and
`get_unix_socket_path` for the local transport used by `pi-client`.

The source package's `testing/` adapters are excluded because they are test
support rather than the runtime architecture. This port is implementation
only; runtime verification remains deferred to the final unified test phase.
