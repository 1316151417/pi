# pi-telemetry

Python port of `@earendil-works/pi-telemetry` 0.85.1. Requires Python 3.12 or newer and has no runtime dependencies. Public methods and fields use snake_case.

## Source mapping

| TypeScript source | Python implementation |
| --- | --- |
| `packages/telemetry/src/index.ts`: context, span, status, schema and inferred-type contracts | `src/pi_telemetry/types.py` |
| `packages/telemetry/src/index.ts`: `defineTelemetrySchema`, `bindTypedSpanStarter`, `createTypedSpanStarter` | `src/pi_telemetry/schema.py` |
| `packages/telemetry/src/index.ts`: public exports | `src/pi_telemetry/__init__.py` |
| `packages/telemetry/src/noop.ts` | `src/pi_telemetry/noop.py` |
| `packages/telemetry/src/memory.ts`: runtime and recording state | `src/pi_telemetry/memory.py` |
| `packages/telemetry/src/memory.ts`: public record interfaces | `src/pi_telemetry/types.py` |
| Promise resolution in `noop.ts` and `memory.ts` | `src/pi_telemetry/_promise.py` |
| `packages/telemetry/src/testing/types.ts` | `src/pi_telemetry/testing/types.py` |
| `packages/telemetry/src/testing/index.ts`: type exports only | `src/pi_telemetry/testing/__init__.py` |

`packages/telemetry/src/testing/conformance.ts` and its `createTelemetryAdapterConformance` export are deferred with all tests, as requested. No test cases, imports, compilation probes, checks, or test commands have been run for this package.

## Runtime behavior

`start_span` synchronously creates a span and invokes its callback exactly once, returning an awaitable. Within an active asyncio loop, callback results are scheduled immediately. Returned values and exception objects pass through unchanged. Explicit status is last-write-wins; automatic error status applies only when no explicit status was set. Attribute updates and events are synchronous, passive, atomic per call, and ignored after settlement. Child spans started through a settled span execute through the shared no-op context. Snapshots copy records, events, mutable attribute lists, statuses, and error details and preserve deterministic start IDs and completion sequence numbers.

`define_telemetry_schema` is an identity helper. `create_typed_span_starter` neither examines nor retains schemas and binds child starters to the callback span. Neither helper adds runtime validation, including duplicate-name or parent checks.

## Python boundaries and pending verification

- Python `None` represents TypeScript's omitted/undefined attribute values and is ignored during attribute updates. Optional record fields such as `end_sequence` use `None` until present.
- Public data shapes use dataclasses and mappings. Use `dataclasses.asdict` when a dictionary representation is needed; schema field names are snake_case.
- Python cannot infer literal attribute keys and per-span overloads from runtime schema objects as TypeScript conditional types do. Corresponding public inference aliases retain structural types without compile-time closed-set checking.
- Outside a running asyncio loop, callback invocation still occurs immediately, while awaitable execution and successful settlement wait until the result is awaited. Python coroutine bodies begin when scheduled; calling an async callback cannot synchronously execute the body as a JavaScript async function does.
- Python supports exception rejection values, rather than arbitrary JavaScript thrown values. Error names use the exception's `name` attribute when supplied, otherwise its class name.
- The testing fixture disposal contract uses Python's asynchronous context manager protocol in place of `Symbol.asyncDispose`.
- Thread safety, type checking, cancellation behavior, snapshot behavior, adapter conformance, and integration with dependent packages remain unverified pending the requested final testing phase.
