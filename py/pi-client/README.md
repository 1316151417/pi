# pi-client

Python port of `packages/client/src` for learning the remote pi session path.
The core `Client` is transport neutral. It uses `pi-protocol` for framed CBOR
messages and `pi-chord` for service calls, catalogue, subscriptions and Delta
state updates. `unix.py` provides a concrete local socket transport and server
discovery.

| TypeScript | Python |
| --- | --- |
| `types.ts`, `transport.ts`, `errors.ts`, `promise.ts` | `types.py`, `transport.py`, `errors.py`, `promise.py` |
| `connection.ts` | `connection.py` |
| `client.ts` | `client.py` |
| `unix.ts` | `unix.py` |

`Client.connect(options)` opens a new connected client; `client.connect()` and
`client.reconnect()` reconnect an existing instance. Python returns an asyncio
future where TypeScript returns a Promise. `client.request(...)` starts the
request immediately and returns a future. The returned service subscription
must call `start()` after installing the snapshot so queued updates are released
in order. `create_client_service_transport(client, get_target)` adapts it to a
Chord remote service binding.

`create_unix_transport_factory(UnixTransportOptions(path=...))` gives the client
a concrete Unix socket transport. `discover_unix_servers(...)` probes canonical
server-addressed socket names with at most 16 concurrent handshakes. The Python
adapter uses `asyncio` streams; `StreamWriter.drain()` provides write backpressure
instead of Node's write callback and `drain` events. Repeated close is harmless.

The protocol, connection, request cancellation, attachment notifications,
subscription hydration, ordered update delivery, disposal, socket writes and
discovery were ported from the current TypeScript source. Python callback and
event-loop scheduling differs from JavaScript microtasks. The source's
`AbortSignal` is represented by Chord's structural `AbortSignalLike` protocol.

This implementation has not been imported, compiled, exercised against a
server, or tested; verification remains in the final unified test stage.
