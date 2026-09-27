# pi-durable

Native Python port of the current `packages/durable/src` public API: durable
record contracts and detached in-memory storage. Python 3.12 or newer is
required. Runtime dependency: `pi-chord==0.85.1`.

| TypeScript source | Python module |
| --- | --- |
| `packages/durable/src/types.ts` | `pi_durable/types.py` |
| `packages/durable/src/memory-storage.ts` | `pi_durable/memory_storage.py` |
| `packages/durable/src/index.ts` | `pi_durable/__init__.py` |

Records use `TypedDict` with the persisted camelCase field names, including
`conversationId`, `abortRequested`, and `commitSeq`. Methods use snake_case.
Optional keys remain absent. Missing lookups return Chord's exported
`UNDEFINED`, which is distinct from JSON `None`. Union variants have separate
named contracts; their combined aliases match the TypeScript declarations.
Python dictionaries remain mutable, matching the source's runtime objects;
readonly ownership is expressed by types and detached storage reads/writes.

```python
from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import MemoryStorage, ROOT_CONVERSATION_ID, UNDEFINED

storage = MemoryStorage()
await storage.commit(
    [{"type": "conversation", "value": {"id": ROOT_CONVERSATION_ID}}],
    BACKGROUND_CONTEXT,
)
entry_id = storage.mint_id()
await storage.commit(
    [{"type": "entry", "value": {
        "id": entry_id, "conversationId": ROOT_CONVERSATION_ID,
        "kind": "note", "data": {"text": "saved"},
    }}],
    BACKGROUND_CONTEXT,
)
page = await storage.scan_entries(
    {"conversationId": ROOT_CONVERSATION_ID}, UNDEFINED, 20, BACKGROUND_CONTEXT,
)
await storage.close(BACKGROUND_CONTEXT)
```

The implementation preserves global cross-table ID ownership, immutable
conversation/entry creation, complete task/input replacement, input request
index replacement, status indexes, ascending conversation/task scans,
newest-first fork-aware entry scans, inclusive cutoffs, head-marker ancestry,
continuation cursors, detached nested values, and the original error messages.
Empty commits still receive a sequence. The root conversation is not inserted
automatically; ID allocation starts at 2. Context parameters are intentionally
unused, as in the source; cancellation does not affect storage operations.

The TypeScript methods contain no internal `await`. The Python methods
therefore execute immediately and return a reusable settled awaitable. A
failure is raised when that result is awaited, while `mint_id` raises directly.
Awaiting the result yields one asyncio scheduling turn. Use `await` or
`asyncio.ensure_future` for these results; they are awaitables rather than
coroutine objects accepted directly by `asyncio.create_task`. Closing takes
effect immediately, repeated closes succeed, and later operations reject.

Cloning deliberately does not preserve shared references between branches
and does not accept cycles, matching the recursive source clone. Python
dictionaries have no JavaScript prototype, so keys such as `__proto__` are
ordinary retained keys. Numeric object keys follow JavaScript enumeration
order. Tuples used for readonly arrays are copied into lists. Native Pi AI
message dataclasses are copied field-by-field without constructors; this
preserves their Python representation while detaching their nested content.
The optional `pi_ai.types.Message` annotation is imported only for static
typing, so memory storage has no runtime dependency on `pi-ai`.

IDs and commit arithmetic follow binary64 number behavior and the original
safe-integer exhaustion guard. Callers should supply semantically valid
records, ancestry, transitions, and backend-generated cursors, as required by
the TypeScript storage contract. No additional validation is introduced.
Scan-limit slicing retains the source behavior, including its error for a
zero-sized page when further records exist.

Implementation has been compared with the full source. No unit tests, runtime
tests, import probes, or compilation checks have been run. Verification is
deferred to the final unified testing phase. Workspace registration and lock
integration are handled separately. Existing agent storage implementations
remain unchanged because they expose different contracts.
