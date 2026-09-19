"""Session repository conformance cases ported from ``session/testing/conformance/session-repo.ts``.

Each exported ``create_*_conformance`` function mirrors the TypeScript factory of
the same name: it returns runner-independent :class:`ConformanceCase` values that
own a fresh repository per run. Group names, case names, and assertions are
reproduced one to one.

Concurrency note: the TypeScript table starts promises eagerly by calling
``repo.fork(...)``/``session.mutate(...)`` before awaiting them. Python coroutines
start when awaited, so cases that need overlapping repository operations wrap
them in ``asyncio.ensure_future`` and combine them with ``asyncio.gather``
(``Promise.all``) or ``asyncio.gather(..., return_exceptions=True)``
(``Promise.allSettled``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from pi_ai.types import AssistantMessage, DeferredHandle, TextContent, ToolCall, Usage, UserMessage
from ....context import BACKGROUND_CONTEXT
from ...commit import insert_entry, insert_usage
from ...types import (
    CustomEntry,
    EntryQuery,
    ForkOptions,
    MessageEntry,
    Session,
    SessionCreateOptions,
    SessionRepo,
    SessionStats,
    UsageRow,
)
from ...values import (
    ListCursor,
    ListElement,
    ListReadOptions,
    append_list,
    branch_tip,
    delete_list,
    entry_label,
    lane_config,
    lane_state,
    list as list_address,
    operation_meta,
    operation_preparation,
    operation_result,
    operation_state,
    operation_tool_args,
    pending_entry,
    session_name,
    set_value,
    value,
)
from ..types import ConformanceCase
from .assertions import deep_strict_equal, rejects, strict_equal

__all__ = [
    "create_session_repo_conformance",
    "create_session_repo_fork_behavior_conformance",
    "create_session_repo_fork_conformance",
    "create_session_repo_fork_coordination_conformance",
    "create_session_repo_fork_destination_reservation_conformance",
    "create_session_repo_fork_source_snapshot_conformance",
    "create_session_repo_lifecycle_conformance",
    "create_session_repo_message_conformance",
    "create_session_repo_ownership_conformance",
    "create_session_repo_streaming_fork_conformance",
]

ROOT_ID = "00000000-0000-7000-8000-000000000001"
CHILD_ID = "00000000-0000-7000-8000-000000000002"
SIBLING_ID = "00000000-0000-7000-8000-000000000003"
USAGE_ID = "00000000-0000-7000-8000-000000000004"
OPERATION_ID = "00000000-0000-7000-8000-000000000005"
PENDING_ID = "00000000-0000-7000-8000-000000000006"
UNKNOWN_ID = "00000000-0000-7000-8000-000000000007"

_APPLICATION_VALUE = value("test.application.value")
_APPLICATION_LIST = list_address("test.application.list")

STOP_REASONS = ("stop", "length", "toolUse", "error", "aborted", "deferred")


def _configuration() -> dict:
    """The lane configuration payload the TypeScript table writes as an object literal.

    Lane configuration and lane state are JSON values on the wire, so this port
    seeds the same JSON shape every backend round-trips.
    """
    return {
        "model": {"provider": "provider", "modelId": "model"},
        "thinkingLevel": "off",
        "activeToolNames": ["read"],
    }


def _idle_lane_state() -> dict:
    """The lane-state payload a fork projects into its destination.

    TypeScript compares against the ``idleLaneState`` object literal; this port
    projects exactly that JSON shape (see ``fork_policy.project_fork_current_state_write``).
    """
    return {"currentOperationId": None, "lastOperationId": None, "inbox": []}


def _is_safe_integer(candidate: Any) -> bool:
    return isinstance(candidate, int) and not isinstance(candidate, bool)


async def _get_branch_tip(session: Session, name: str = "main") -> Optional[str]:
    branch = await session.branch(name, BACKGROUND_CONTEXT)
    return None if branch is None else await branch.get_tip_id(BACKGROUND_CONTEXT)


def _assistant_message(stop_reason: str) -> AssistantMessage:
    content = (
        [ToolCall(id="call", name="read", arguments={})]
        if stop_reason == "toolUse"
        else [TextContent(text=stop_reason)]
    )
    return AssistantMessage(
        content=content,
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(),
        stop_reason=stop_reason,
        deferred=(
            DeferredHandle(
                provider="anthropic",
                model_id="claude-sonnet-4-5",
                api="anthropic-messages",
                id="job",
            )
            if stop_reason == "deferred"
            else None
        ),
        timestamp=1,
    )


def _usage_row() -> UsageRow:
    return UsageRow(id=USAGE_ID, adjustment=True, usage=Usage(input=1, output=2, total_tokens=3))


@dataclass
class _RepoCaseContext:
    """One prepared repository plus the optional cleanup the factory registered."""

    repo: SessionRepo
    close: Optional[Callable[[], Any]] = None


def _prepare_repo_case_factory(
    factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> Callable[[], Awaitable[_RepoCaseContext]]:
    async def prepared() -> _RepoCaseContext:
        return _RepoCaseContext(repo=await factory(), close=on_close)

    return prepared


def _create_case(
    factory: Callable[[], Awaitable[_RepoCaseContext]],
    group: str,
    name: str,
    test: Callable[[_RepoCaseContext], Awaitable[None]],
) -> ConformanceCase:
    async def run() -> None:
        context = await factory()
        try:
            await test(context)
        finally:
            if context.close is not None:
                closing = context.close()
                if asyncio.iscoroutine(closing) or asyncio.isfuture(closing):
                    await closing

    return ConformanceCase(group=group, name=name, run=run)


def _create_repo_commit(
    writes: List[Any],
) -> Callable[[Any, Any], Awaitable[None]]:
    """Build a ``mutate`` callback that commits ``writes`` and propagates failures."""

    async def commit(mutator: Any, mutation_context: Any) -> Any:
        return await mutator.commit(writes, mutation_context)

    return commit


def create_session_repo_lifecycle_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create lifecycle cases for repositories that support creation, discovery, open, and deletion."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def creates_session_without_branch(context: _RepoCaseContext) -> None:
        repo = context.repo
        session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)

        strict_equal(session.metadata.id, "session")
        strict_equal(_is_safe_integer(session.metadata.created_at), True)
        strict_equal(session.metadata.storage_version, 1)
        strict_equal(await session.branch("main", BACKGROUND_CONTEXT), None)
        strict_equal(await session.get_value(lane_state("main"), BACKGROUND_CONTEXT), None)
        strict_equal(await session.get_value(lane_config("main"), BACKGROUND_CONTEXT), None)
        await rejects(repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT))
        await session.close(BACKGROUND_CONTEXT)

    async def close_drains_scope(context: _RepoCaseContext) -> None:
        repo = context.repo
        session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
        active = await session.begin_mutation(BACKGROUND_CONTEXT)
        queued_started = False

        async def queued_callback(_mutator: Any, _mutation_context: Any) -> None:
            nonlocal queued_started
            queued_started = True

        queued = asyncio.ensure_future(session.mutate(queued_callback, BACKGROUND_CONTEXT))
        closing = asyncio.ensure_future(session.close(BACKGROUND_CONTEXT))

        await active.end(BACKGROUND_CONTEXT)
        await rejects(queued)
        await closing
        strict_equal(queued_started, False)

    async def lists_metadata_and_reopens(context: _RepoCaseContext) -> None:
        repo = context.repo
        first = await repo.create(SessionCreateOptions(id="first"), BACKGROUND_CONTEXT)
        await first.set_name("preserved", BACKGROUND_CONTEXT)
        second = await repo.create(
            SessionCreateOptions(id="second", parent_session_id="parent"), BACKGROUND_CONTEXT
        )
        await second.close(BACKGROUND_CONTEXT)

        listed = await repo.list(None, BACKGROUND_CONTEXT)
        deep_strict_equal(
            sorted((metadata.id, metadata.parent_session_id) for metadata in listed),
            [("first", None), ("second", "parent")],
        )
        await first.close(BACKGROUND_CONTEXT)
        await rejects(first.get_name(BACKGROUND_CONTEXT))
        reopened = await repo.open(first.metadata, BACKGROUND_CONTEXT)
        strict_equal(reopened is first, False)
        strict_equal(await reopened.get_name(BACKGROUND_CONTEXT), "preserved")
        await reopened.close(BACKGROUND_CONTEXT)

    async def deletes_closed_sessions(context: _RepoCaseContext) -> None:
        repo = context.repo
        removed = await repo.create(SessionCreateOptions(id="removed"), BACKGROUND_CONTEXT)
        retained = await repo.create(SessionCreateOptions(id="retained"), BACKGROUND_CONTEXT)
        await asyncio.gather(removed.close(BACKGROUND_CONTEXT), retained.close(BACKGROUND_CONTEXT))

        await repo.delete(removed.metadata, BACKGROUND_CONTEXT)
        deep_strict_equal(
            [metadata.id for metadata in await repo.list(None, BACKGROUND_CONTEXT)], ["retained"]
        )
        await rejects(repo.open(removed.metadata, BACKGROUND_CONTEXT))
        await rejects(repo.delete(removed.metadata, BACKGROUND_CONTEXT))

    return [
        _create_case(
            factory,
            "lifecycle",
            "creates a session with no implicit branch and rejects duplicate ids",
            creates_session_without_branch,
        ),
        _create_case(
            factory,
            "lifecycle",
            "close drains an acquired scope and rejects a queued mutation callback",
            close_drains_scope,
        ),
        _create_case(
            factory,
            "lifecycle",
            "lists metadata and preserves state across close and reopen",
            lists_metadata_and_reopens,
        ),
        _create_case(
            factory,
            "lifecycle",
            "deletes closed sessions without affecting other sessions",
            deletes_closed_sessions,
        ),
    ]


def create_session_repo_ownership_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create exclusive-open cases for repositories that own active session handles."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def rejects_opening_open_session(context: _RepoCaseContext) -> None:
        repo = context.repo
        session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
        await rejects(repo.open(session.metadata, BACKGROUND_CONTEXT))
        await session.close(BACKGROUND_CONTEXT)

        reopened = await repo.open(session.metadata, BACKGROUND_CONTEXT)
        await rejects(repo.open(session.metadata, BACKGROUND_CONTEXT))
        await reopened.close(BACKGROUND_CONTEXT)

    return [
        _create_case(
            factory, "ownership", "rejects opening an already-open session", rejects_opening_open_session
        )
    ]


def create_session_repo_message_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create message cases for repositories that support session creation."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def rejects_pending_assistant(context: _RepoCaseContext) -> None:
        repo = context.repo
        session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
        branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)

        await rejects(branch.append_message(_assistant_message("pending"), BACKGROUND_CONTEXT))

        strict_equal(await _get_branch_tip(session), None)
        deep_strict_equal(await session.find_entries(None, BACKGROUND_CONTEXT), [])
        await session.close(BACKGROUND_CONTEXT)

    async def preserves_settled_stop_reasons(context: _RepoCaseContext) -> None:
        repo = context.repo
        session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
        messages = [_assistant_message(stop_reason) for stop_reason in STOP_REASONS]
        branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
        ids: List[str] = []

        for message in messages:
            ids.append(await branch.append_message(message, BACKGROUND_CONTEXT))

        entries = await session.find_entries(
            EntryQuery(order="asc", type="message"), BACKGROUND_CONTEXT
        )
        deep_strict_equal([entry.id for entry in entries], ids)
        for index, entry in enumerate(entries):
            deep_strict_equal(entry.message, messages[index])
        strict_equal(await _get_branch_tip(session), ids[-1])
        await session.close(BACKGROUND_CONTEXT)

    return [
        _create_case(
            factory,
            "messages",
            "rejects pending assistant messages without changing the tree",
            rejects_pending_assistant,
        ),
        _create_case(
            factory,
            "messages",
            "preserves every settled assistant stop reason",
            preserves_settled_stop_reasons,
        ),
    ]


def create_session_repo_fork_behavior_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create fork-content cases that do not require concurrent repository coordination."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def tree_forks_fresh_session(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        fork = await repo.fork(source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT)

        strict_equal(fork.metadata.id, "fork")
        strict_equal(fork.metadata.parent_session_id, "source")
        strict_equal(await fork.branch("main", BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(lane_config("main"), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(lane_state("main"), BACKGROUND_CONTEXT), None)
        deep_strict_equal(await fork.find_entries(None, BACKGROUND_CONTEXT), [])
        deep_strict_equal(
            await fork.get_stats(BACKGROUND_CONTEXT),
            SessionStats(message_count=0, usage=Usage()),
        )
        await asyncio.gather(source.close(BACKGROUND_CONTEXT), fork.close(BACKGROUND_CONTEXT))

    async def rejects_data_only_branch(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.create_branch("data", None, BACKGROUND_CONTEXT)

        await rejects(
            repo.fork(
                source.metadata,
                ForkOptions(id="destination", scope="branch", branch="data"),
                BACKGROUND_CONTEXT,
            )
        )
        deep_strict_equal(
            [metadata.id for metadata in await repo.list(None, BACKGROUND_CONTEXT)], ["source"]
        )

        destination = await repo.create(SessionCreateOptions(id="destination"), BACKGROUND_CONTEXT)
        await asyncio.gather(source.close(BACKGROUND_CONTEXT), destination.close(BACKGROUND_CONTEXT))

    async def forks_configured_branch(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.mutate(
            _create_repo_commit(
                [
                    insert_entry(CustomEntry(id=ROOT_ID, parent_id=None, custom_type="root")),
                    insert_entry(
                        MessageEntry(
                            id=CHILD_ID,
                            parent_id=ROOT_ID,
                            message=UserMessage(content="child", timestamp=1),
                        )
                    ),
                    insert_entry(CustomEntry(id=SIBLING_ID, parent_id=ROOT_ID, custom_type="sibling")),
                    set_value(branch_tip("main"), SIBLING_ID),
                    set_value(branch_tip("review"), CHILD_ID),
                    set_value(lane_config("review"), _configuration()),
                    set_value(
                        lane_state("review"),
                        {
                            "currentOperationId": OPERATION_ID,
                            "lastOperationId": "previous",
                            "inbox": [{"entryId": PENDING_ID, "kind": "write"}],
                        },
                    ),
                    set_value(
                        operation_result("previous"),
                        {
                            "operationId": "previous",
                            "kind": "navigation",
                            "status": "completed",
                            "fromTipId": ROOT_ID,
                            "tipId": CHILD_ID,
                            "startedAt": 1,
                            "endedAt": 2,
                        },
                    ),
                    set_value(session_name(), "source name"),
                    set_value(_APPLICATION_VALUE, {"copied": False}),
                    append_list(_APPLICATION_LIST, {"copied": False}),
                    set_value(entry_label(ROOT_ID), "root label"),
                    set_value(entry_label(SIBLING_ID), "sibling label"),
                    set_value(pending_entry(PENDING_ID), {"type": "custom", "customType": "pending"}),
                    set_value(
                        operation_meta(OPERATION_ID),
                        {
                            "operationId": OPERATION_ID,
                            "lane": "review",
                            "sourceTipId": CHILD_ID,
                            "startedAt": 1,
                            "intent": {"kind": "compaction"},
                        },
                    ),
                    set_value(
                        operation_state(OPERATION_ID),
                        {
                            "at": "summary.deciding",
                            "control": {"status": "running"},
                            "settings": {
                                "compaction": {"enabled": True, "reserveTokens": 1, "keepRecentTokens": 1},
                                "steeringMode": "all",
                                "followUpMode": "all",
                                "toolExecution": "sequential",
                            },
                            "latestAssistantEntryId": None,
                            "task": {
                                "taskId": OPERATION_ID,
                                "reason": "manual",
                                "boundary": {"kind": "finish"},
                            },
                        },
                    ),
                    set_value(operation_tool_args(OPERATION_ID, ROOT_ID, 0), {"argument": True}),
                    set_value(
                        operation_preparation(OPERATION_ID, OPERATION_ID),
                        {
                            "kind": "compaction",
                            "messagesToSummarize": [],
                            "turnPrefixMessages": [],
                            "retainedTail": [],
                            "isSplitTurn": False,
                            "tokensBefore": 0,
                            "fileOps": {"read": [], "written": [], "edited": []},
                            "settings": {"enabled": True, "reserveTokens": 1, "keepRecentTokens": 1},
                        },
                    ),
                    insert_usage(_usage_row()),
                ]
            ),
            BACKGROUND_CONTEXT,
        )

        fork = await repo.fork(
            source.metadata,
            ForkOptions(
                id="fork", scope="branch", branch="review", entry_id=CHILD_ID, position="at"
            ),
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            [entry.id for entry in await fork.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)],
            [ROOT_ID, CHILD_ID],
        )
        strict_equal(await fork.branch("main", BACKGROUND_CONTEXT), None)
        strict_equal(await _get_branch_tip(fork, "review"), CHILD_ID)
        stored_config = await fork.get_value(lane_config("review"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_config is None else stored_config.value, _configuration())
        stored_state = await fork.get_value(lane_state("review"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_state is None else stored_state.value, _idle_lane_state())
        strict_equal(await fork.get_value(lane_config("main"), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(lane_state("main"), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_name(BACKGROUND_CONTEXT), "source name")
        strict_equal(await fork.get_value(_APPLICATION_VALUE, BACKGROUND_CONTEXT), None)
        deep_strict_equal(await fork.read_list(_APPLICATION_LIST, None, BACKGROUND_CONTEXT), [])
        strict_equal(await fork.get_label(ROOT_ID, BACKGROUND_CONTEXT), "root label")
        strict_equal(await fork.get_label(SIBLING_ID, BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(operation_result("previous"), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(pending_entry(PENDING_ID), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(operation_meta(OPERATION_ID), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(operation_state(OPERATION_ID), BACKGROUND_CONTEXT), None)
        strict_equal(
            await fork.get_value(operation_tool_args(OPERATION_ID, ROOT_ID, 0), BACKGROUND_CONTEXT),
            None,
        )
        strict_equal(
            await fork.get_value(
                operation_preparation(OPERATION_ID, OPERATION_ID), BACKGROUND_CONTEXT
            ),
            None,
        )
        stats = await fork.get_stats(BACKGROUND_CONTEXT)
        strict_equal(stats.message_count, 1)
        deep_strict_equal(stats.usage, Usage())
        await asyncio.gather(source.close(BACKGROUND_CONTEXT), fork.close(BACKGROUND_CONTEXT))

    async def enforces_branch_ancestry(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.mutate(
            _create_repo_commit(
                [
                    insert_entry(CustomEntry(id=ROOT_ID, parent_id=None, custom_type="root")),
                    insert_entry(CustomEntry(id=CHILD_ID, parent_id=ROOT_ID, custom_type="child")),
                    insert_entry(CustomEntry(id=SIBLING_ID, parent_id=ROOT_ID, custom_type="sibling")),
                    set_value(branch_tip("main"), CHILD_ID),
                    set_value(lane_config("main"), _configuration()),
                    set_value(lane_state("main"), _idle_lane_state()),
                    set_value(branch_tip("empty"), None),
                    set_value(lane_config("empty"), _configuration()),
                    set_value(lane_state("empty"), _idle_lane_state()),
                ]
            ),
            BACKGROUND_CONTEXT,
        )

        before = await repo.fork(
            source.metadata,
            ForkOptions(id="before", scope="branch", branch="main", entry_id=CHILD_ID, position="before"),
            BACKGROUND_CONTEXT,
        )
        strict_equal(await _get_branch_tip(before), ROOT_ID)
        deep_strict_equal(
            [entry.id for entry in await before.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)],
            [ROOT_ID],
        )

        mid = await repo.fork(
            source.metadata,
            ForkOptions(id="mid", scope="branch", branch="main", entry_id=ROOT_ID, position="at"),
            BACKGROUND_CONTEXT,
        )
        strict_equal(await _get_branch_tip(mid), ROOT_ID)
        deep_strict_equal(
            [entry.id for entry in await mid.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)],
            [ROOT_ID],
        )

        before_root = await repo.fork(
            source.metadata,
            ForkOptions(
                id="before-root", scope="branch", branch="main", entry_id=ROOT_ID, position="before"
            ),
            BACKGROUND_CONTEXT,
        )
        strict_equal(await _get_branch_tip(before_root), None)
        deep_strict_equal(
            await before_root.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT), []
        )

        empty = await repo.fork(
            source.metadata,
            ForkOptions(id="empty", scope="branch", branch="empty"),
            BACKGROUND_CONTEXT,
        )
        strict_equal(await _get_branch_tip(empty, "empty"), None)
        deep_strict_equal(await empty.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT), [])

        for entry_id, branch, requested in [
            ("off-branch", "main", SIBLING_ID),
            ("unknown", "main", UNKNOWN_ID),
            ("null-tip", "empty", ROOT_ID),
        ]:
            await rejects(
                repo.fork(
                    source.metadata,
                    ForkOptions(id=entry_id, scope="branch", branch=branch, entry_id=requested),
                    BACKGROUND_CONTEXT,
                )
            )
        deep_strict_equal(
            sorted(metadata.id for metadata in await repo.list(None, BACKGROUND_CONTEXT)),
            ["before", "before-root", "empty", "mid", "source"],
        )
        await asyncio.gather(
            source.close(BACKGROUND_CONTEXT),
            before.close(BACKGROUND_CONTEXT),
            before_root.close(BACKGROUND_CONTEXT),
            empty.close(BACKGROUND_CONTEXT),
            mid.close(BACKGROUND_CONTEXT),
        )

    async def forks_closed_source(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.mutate(
            _create_repo_commit(
                [
                    insert_entry(CustomEntry(id=ROOT_ID, parent_id=None, custom_type="root")),
                    set_value(branch_tip("main"), ROOT_ID),
                    set_value(lane_config("main"), _configuration()),
                    set_value(lane_state("main"), _idle_lane_state()),
                    set_value(_APPLICATION_VALUE, "excluded"),
                    append_list(_APPLICATION_LIST, "excluded"),
                ]
            ),
            BACKGROUND_CONTEXT,
        )
        await source.close(BACKGROUND_CONTEXT)

        fork = await repo.fork(
            source.metadata,
            ForkOptions(id="fork", scope="branch", branch="main"),
            BACKGROUND_CONTEXT,
        )
        strict_equal(await _get_branch_tip(fork), ROOT_ID)
        stored_config = await fork.get_value(lane_config("main"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_config is None else stored_config.value, _configuration())
        stored_state = await fork.get_value(lane_state("main"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_state is None else stored_state.value, _idle_lane_state())
        strict_equal(await fork.get_value(_APPLICATION_VALUE, BACKGROUND_CONTEXT), None)
        deep_strict_equal(await fork.read_list(_APPLICATION_LIST, None, BACKGROUND_CONTEXT), [])
        await fork.close(BACKGROUND_CONTEXT)

    async def forks_whole_configured_tree(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.mutate(
            _create_repo_commit(
                [
                    insert_entry(CustomEntry(id=ROOT_ID, parent_id=None, custom_type="root")),
                    insert_entry(CustomEntry(id=CHILD_ID, parent_id=ROOT_ID, custom_type="child")),
                    insert_entry(CustomEntry(id=SIBLING_ID, parent_id=ROOT_ID, custom_type="sibling")),
                    set_value(branch_tip("main"), CHILD_ID),
                    set_value(lane_config("main"), _configuration()),
                    set_value(lane_state("main"), _idle_lane_state()),
                    set_value(branch_tip("review"), SIBLING_ID),
                    set_value(lane_config("review"), _configuration()),
                    set_value(lane_state("review"), _idle_lane_state()),
                    set_value(branch_tip("notes"), ROOT_ID),
                    set_value(_APPLICATION_VALUE, {"copied": True}),
                ]
            ),
            BACKGROUND_CONTEXT,
        )

        fork = await repo.fork(source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT)

        deep_strict_equal(
            [entry.id for entry in await fork.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)],
            [ROOT_ID, CHILD_ID, SIBLING_ID],
        )
        strict_equal(await _get_branch_tip(fork), CHILD_ID)
        strict_equal(await _get_branch_tip(fork, "review"), SIBLING_ID)
        strict_equal(await _get_branch_tip(fork, "notes"), ROOT_ID)
        strict_equal(await fork.get_value(lane_config("notes"), BACKGROUND_CONTEXT), None)
        strict_equal(await fork.get_value(lane_state("notes"), BACKGROUND_CONTEXT), None)
        stored_config = await fork.get_value(lane_config("review"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_config is None else stored_config.value, _configuration())
        stored_state = await fork.get_value(lane_state("review"), BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_state is None else stored_state.value, _idle_lane_state())
        stored_value = await fork.get_value(_APPLICATION_VALUE, BACKGROUND_CONTEXT)
        deep_strict_equal(None if stored_value is None else stored_value.value, {"copied": True})
        await asyncio.gather(source.close(BACKGROUND_CONTEXT), fork.close(BACKGROUND_CONTEXT))

    async def rejects_unknown_reserved_state(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        await source.create_branch("main", None, BACKGROUND_CONTEXT)
        await source.mutate(
            _create_repo_commit(
                [
                    set_value(lane_config("main"), _configuration()),
                    set_value(lane_state("main"), _idle_lane_state()),
                ]
            ),
            BACKGROUND_CONTEXT,
        )

        for namespace in ("pi", "pi.unknown"):
            address = value(namespace)
            await source.set_value(address, True, BACKGROUND_CONTEXT)
            await rejects(
                repo.fork(source.metadata, ForkOptions(id="tree", scope="tree"), BACKGROUND_CONTEXT)
            )
            await rejects(
                repo.fork(
                    source.metadata,
                    ForkOptions(id="branch", scope="branch", branch="main"),
                    BACKGROUND_CONTEXT,
                )
            )
            await source.delete_value(address, BACKGROUND_CONTEXT)
        await source.close(BACKGROUND_CONTEXT)

        tree = await repo.fork(source.metadata, ForkOptions(id="tree", scope="tree"), BACKGROUND_CONTEXT)
        branch = await repo.fork(
            source.metadata,
            ForkOptions(id="branch", scope="branch", branch="main"),
            BACKGROUND_CONTEXT,
        )
        await asyncio.gather(tree.close(BACKGROUND_CONTEXT), branch.close(BACKGROUND_CONTEXT))

    return [
        _create_case(
            factory, "forks", "tree-forks a fresh session before first attachment", tree_forks_fresh_session
        ),
        _create_case(
            factory,
            "forks",
            "rejects a data-only branch and releases its destination id",
            rejects_data_only_branch,
        ),
        _create_case(
            factory,
            "forks",
            "forks one named configured branch with scoped values and a zero ledger",
            forks_configured_branch,
        ),
        _create_case(
            factory,
            "forks",
            "enforces branch ancestry for at and before placement",
            enforces_branch_ancestry,
        ),
        _create_case(factory, "forks", "forks a closed source session", forks_closed_source),
        _create_case(
            factory, "forks", "forks the whole configured tree with fresh lane state", forks_whole_configured_tree
        ),
        _create_case(
            factory,
            "forks",
            "rejects only surviving unknown reserved scalar state",
            rejects_unknown_reserved_state,
        ),
    ]


def create_session_repo_streaming_fork_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """WP08 cases enabled for Memory and JSONL while SQLite's streaming fork implementation is pending."""
    return [
        *_create_session_repo_fork_application_list_conformance(backend_factory, on_close),
        *_create_session_repo_branch_fork_application_state_conformance(backend_factory, on_close),
        *_create_session_repo_fork_lane_validation_conformance(backend_factory, on_close),
    ]


def _create_session_repo_fork_lane_validation_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def ignores_malformed_unrelated_lanes(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        tree: Optional[Session] = None
        branch: Optional[Session] = None
        try:
            await source.create_branch("main", None, BACKGROUND_CONTEXT)
            await source.mutate(
                _create_repo_commit(
                    [
                        set_value(lane_config("main"), _configuration()),
                        set_value(lane_state("main"), _idle_lane_state()),
                        set_value(lane_config("unrelated"), _configuration()),
                    ]
                ),
                BACKGROUND_CONTEXT,
            )

            tree = await repo.fork(source.metadata, ForkOptions(id="tree", scope="tree"), BACKGROUND_CONTEXT)
            branch = await repo.fork(
                source.metadata,
                ForkOptions(id="branch", scope="branch", branch="main"),
                BACKGROUND_CONTEXT,
            )
            strict_equal(await _get_branch_tip(branch), None)
        finally:
            closers = [source.close(BACKGROUND_CONTEXT)]
            if tree is not None:
                closers.append(tree.close(BACKGROUND_CONTEXT))
            if branch is not None:
                closers.append(branch.close(BACKGROUND_CONTEXT))
            await asyncio.gather(*closers)

    return [
        _create_case(
            factory, "fork lane validation", "ignores malformed unrelated lanes", ignores_malformed_unrelated_lanes
        )
    ]


def _create_session_repo_fork_application_list_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    factory = _prepare_repo_case_factory(backend_factory, on_close)
    cases: List[ConformanceCase] = []

    for source_state in ("open", "closed"):
        group = f"fork application lists ({source_state} source)"

        async def copies_lists_at_distinct_addresses(
            context: _RepoCaseContext, source_state: str = source_state
        ) -> None:
            # Each list keeps its own contents: keys "" and "other" in one namespace must not merge.
            # A different namespace also stays separate; pi2.events is application-owned, not reserved.
            repo = context.repo
            source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
            fork: Optional[Session] = None
            try:
                events = list_address("test.application.events")
                sibling = list_address(events.namespace, "other")
                other_namespace = list_address("pi2.events")
                absent = list_address(events.namespace, "absent")
                await source.append_list(events, "event", BACKGROUND_CONTEXT)
                await source.append_list(sibling, "sibling", BACKGROUND_CONTEXT)
                await source.append_list(other_namespace, "other namespace", BACKGROUND_CONTEXT)

                if source_state == "closed":
                    await source.close(BACKGROUND_CONTEXT)
                fork = await repo.fork(
                    source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT
                )

                deep_strict_equal(
                    [
                        element.value
                        for element in await fork.read_list(events, None, BACKGROUND_CONTEXT)
                    ],
                    ["event"],
                )
                deep_strict_equal(
                    [
                        element.value
                        for element in await fork.read_list(sibling, None, BACKGROUND_CONTEXT)
                    ],
                    ["sibling"],
                )
                deep_strict_equal(
                    [
                        element.value
                        for element in await fork.read_list(other_namespace, None, BACKGROUND_CONTEXT)
                    ],
                    ["other namespace"],
                )
                deep_strict_equal(await fork.read_list(absent, None, BACKGROUND_CONTEXT), [])
            finally:
                closers = [source.close(BACKGROUND_CONTEXT)]
                if fork is not None:
                    closers.append(fork.close(BACKGROUND_CONTEXT))
                await asyncio.gather(*closers)

        async def copies_only_survivors(
            context: _RepoCaseContext, source_state: str = source_state
        ) -> None:
            # Append old -> delete -> append new must copy only new, even when changes share a transaction.
            # A list deleted without a later append must remain empty in the fork.
            repo = context.repo
            source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
            fork: Optional[Session] = None
            try:
                events = list_address("test.application.events")
                deleted = list_address(events.namespace, "deleted")
                await source.append_list(events, "old", BACKGROUND_CONTEXT)
                await source.append_list(deleted, "removed", BACKGROUND_CONTEXT)
                await source.mutate(
                    _create_repo_commit(
                        [
                            delete_list(events),
                            append_list(events, "temporary"),
                            delete_list(events),
                            append_list(events, "first survivor"),
                            delete_list(deleted),
                        ]
                    ),
                    BACKGROUND_CONTEXT,
                )
                await source.append_list(events, "second survivor", BACKGROUND_CONTEXT)

                if source_state == "closed":
                    await source.close(BACKGROUND_CONTEXT)
                fork = await repo.fork(
                    source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT
                )

                deep_strict_equal(
                    [
                        element.value
                        for element in await fork.read_list(events, None, BACKGROUND_CONTEXT)
                    ],
                    ["first survivor", "second survivor"],
                )
                deep_strict_equal(await fork.read_list(deleted, None, BACKGROUND_CONTEXT), [])
            finally:
                closers = [source.close(BACKGROUND_CONTEXT)]
                if fork is not None:
                    closers.append(fork.close(BACKGROUND_CONTEXT))
                await asyncio.gather(*closers)

        async def preserves_element_sequences(
            context: _RepoCaseContext, source_state: str = source_state
        ) -> None:
            # Other writes consume sequences too. List elements at seq 2 and 4 must stay at 2 and 4
            # in the fork, not be renumbered to 1 and 2.
            repo = context.repo
            source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
            fork: Optional[Session] = None
            try:
                events = list_address("test.application.events")
                committed = await source.mutate(
                    _create_repo_commit(
                        [
                            set_value(session_name(), "before"),
                            append_list(events, "first"),
                            set_value(session_name(), "between"),
                            append_list(events, "second"),
                        ]
                    ),
                    BACKGROUND_CONTEXT,
                )

                if source_state == "closed":
                    await source.close(BACKGROUND_CONTEXT)
                fork = await repo.fork(
                    source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT
                )

                deep_strict_equal(
                    await fork.read_list(events, None, BACKGROUND_CONTEXT),
                    [
                        ListElement(seq=committed.seqs[1], value="first"),
                        ListElement(seq=committed.seqs[3], value="second"),
                    ],
                )
            finally:
                closers = [source.close(BACKGROUND_CONTEXT)]
                if fork is not None:
                    closers.append(fork.close(BACKGROUND_CONTEXT))
                await asyncio.gather(*closers)

        def continues_pagination(order: str) -> Callable[..., Awaitable[None]]:
            async def continues_pagination_case(
                context: _RepoCaseContext, source_state: str = source_state, order: str = order
            ) -> None:
                # A cursor from the source must resume at the same place in the fork. After reading
                # [first, second] ascending, continue with [third]; descending does the reverse.
                # Continuing after the final element must return an empty page, not repeat that element.
                repo = context.repo
                source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
                fork: Optional[Session] = None
                try:
                    events = list_address("test.application.events")
                    for item in ("first", "second", "third"):
                        await source.append_list(events, item, BACKGROUND_CONTEXT)
                    first_page = await source.read_list(
                        events, ListReadOptions(order=order, limit=2), BACKGROUND_CONTEXT
                    )
                    strict_equal(len(first_page), 2)
                    cursor = ListCursor(seq=first_page[1].seq)
                    last_page = await source.read_list(
                        events, ListReadOptions(order=order, cursor=cursor, limit=2), BACKGROUND_CONTEXT
                    )
                    deep_strict_equal(
                        [element.value for element in last_page],
                        ["third" if order == "asc" else "first"],
                    )
                    end_cursor = ListCursor(seq=last_page[0].seq)

                    if source_state == "closed":
                        await source.close(BACKGROUND_CONTEXT)
                    fork = await repo.fork(
                        source.metadata, ForkOptions(id="fork", scope="tree"), BACKGROUND_CONTEXT
                    )

                    deep_strict_equal(
                        await fork.read_list(
                            events, ListReadOptions(order=order, limit=2), BACKGROUND_CONTEXT
                        ),
                        first_page,
                    )
                    deep_strict_equal(
                        await fork.read_list(
                            events, ListReadOptions(order=order, cursor=cursor, limit=2), BACKGROUND_CONTEXT
                        ),
                        last_page,
                    )
                    deep_strict_equal(
                        await fork.read_list(
                            events,
                            ListReadOptions(order=order, cursor=end_cursor, limit=2),
                            BACKGROUND_CONTEXT,
                        ),
                        [],
                    )
                finally:
                    closers = [source.close(BACKGROUND_CONTEXT)]
                    if fork is not None:
                        closers.append(fork.close(BACKGROUND_CONTEXT))
                    await asyncio.gather(*closers)

            return continues_pagination_case

        cases.extend(
            [
                _create_case(
                    factory,
                    group,
                    "tree fork copies lists at distinct addresses",
                    copies_lists_at_distinct_addresses,
                ),
                _create_case(
                    factory,
                    group,
                    "tree fork copies only survivors after list deletion and reappend",
                    copies_only_survivors,
                ),
                _create_case(
                    factory,
                    group,
                    "tree fork preserves list element sequences including gaps",
                    preserves_element_sequences,
                ),
            ]
        )
        for order in ("asc", "desc"):
            cases.append(
                _create_case(
                    factory,
                    group,
                    f"tree fork continues {order} pagination using source cursors",
                    continues_pagination(order),
                )
            )

    return cases


def _create_session_repo_branch_fork_application_state_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    factory = _prepare_repo_case_factory(backend_factory, on_close)
    cases: List[ConformanceCase] = []

    for source_state in ("open", "closed"):
        group = f"branch fork application state ({source_state} source)"

        async def excludes_overwritten_values(
            context: _RepoCaseContext, source_state: str = source_state
        ) -> None:
            # WP08 §1.1: set v1 -> fork point -> overwrite with v2. A branch fork copies neither
            # version: it cannot reconstruct v1 and must not copy v2. Even unchanged older values
            # are excluded, so filtering current values by the fork point's sequence is not enough.
            repo = context.repo
            source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
            fork: Optional[Session] = None
            try:
                state = value("test.application.state")
                unchanged = value("test.application.settings")
                branch = await source.create_branch("review", None, BACKGROUND_CONTEXT)
                await source.mutate(
                    _create_repo_commit(
                        [
                            set_value(lane_config("review"), _configuration()),
                            set_value(lane_state("review"), _idle_lane_state()),
                            set_value(state, "v1"),
                            set_value(unchanged, "predates fork point"),
                        ]
                    ),
                    BACKGROUND_CONTEXT,
                )
                entry_id = await branch.append_custom_entry("fork-point", None, BACKGROUND_CONTEXT)
                await source.set_value(state, "v2", BACKGROUND_CONTEXT)
                current = await source.get_value(state, BACKGROUND_CONTEXT)
                strict_equal(None if current is None else current.value, "v2")

                if source_state == "closed":
                    await source.close(BACKGROUND_CONTEXT)
                fork = await repo.fork(
                    source.metadata,
                    ForkOptions(scope="branch", branch="review", entry_id=entry_id),
                    BACKGROUND_CONTEXT,
                )

                strict_equal(await _get_branch_tip(fork, "review"), entry_id)
                strict_equal(await fork.get_value(state, BACKGROUND_CONTEXT), None)
                # A survivor older than the fork point must also be excluded, not copied by a seq cutoff.
                strict_equal(await fork.get_value(unchanged, BACKGROUND_CONTEXT), None)
            finally:
                closers = [source.close(BACKGROUND_CONTEXT)]
                if fork is not None:
                    closers.append(fork.close(BACKGROUND_CONTEXT))
                await asyncio.gather(*closers)

        async def excludes_deleted_and_untouched_lists(
            context: _RepoCaseContext, source_state: str = source_state
        ) -> None:
            # WP08 §1.1: append old -> fork point -> delete -> append new. A branch fork copies
            # neither the deleted elements nor the new ones. A separate untouched list is also
            # excluded, even though its elements still survive from before the fork point.
            repo = context.repo
            source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
            fork: Optional[Session] = None
            try:
                events = list_address("test.application.events")
                untouched = list_address(events.namespace, "untouched")
                branch = await source.create_branch("review", None, BACKGROUND_CONTEXT)
                await source.mutate(
                    _create_repo_commit(
                        [
                            set_value(lane_config("review"), _configuration()),
                            set_value(lane_state("review"), _idle_lane_state()),
                            append_list(events, "old first"),
                            append_list(events, "old second"),
                            append_list(untouched, "predates fork point"),
                        ]
                    ),
                    BACKGROUND_CONTEXT,
                )
                entry_id = await branch.append_custom_entry("fork-point", None, BACKGROUND_CONTEXT)
                await source.delete_list(events, BACKGROUND_CONTEXT)
                await source.append_list(events, "new", BACKGROUND_CONTEXT)
                deep_strict_equal(
                    [
                        element.value
                        for element in await source.read_list(events, None, BACKGROUND_CONTEXT)
                    ],
                    ["new"],
                )

                if source_state == "closed":
                    await source.close(BACKGROUND_CONTEXT)
                fork = await repo.fork(
                    source.metadata,
                    ForkOptions(scope="branch", branch="review", entry_id=entry_id),
                    BACKGROUND_CONTEXT,
                )

                strict_equal(await _get_branch_tip(fork, "review"), entry_id)
                deep_strict_equal(await fork.read_list(events, None, BACKGROUND_CONTEXT), [])
                # Even elements still surviving from before the fork point are excluded.
                deep_strict_equal(await fork.read_list(untouched, None, BACKGROUND_CONTEXT), [])
            finally:
                closers = [source.close(BACKGROUND_CONTEXT)]
                if fork is not None:
                    closers.append(fork.close(BACKGROUND_CONTEXT))
                await asyncio.gather(*closers)

        cases.extend(
            [
                _create_case(
                    factory,
                    group,
                    "excludes overwritten and unchanged application values",
                    excludes_overwritten_values,
                ),
                _create_case(
                    factory,
                    group,
                    "excludes deleted/reappended and untouched application lists",
                    excludes_deleted_and_untouched_lists,
                ),
            ]
        )

    return cases


def create_session_repo_fork_destination_reservation_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create fork cases that require destination reservation across create and fork."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def create_reserves_first(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        results = await asyncio.gather(
            repo.create(SessionCreateOptions(id="destination"), BACKGROUND_CONTEXT),
            repo.fork(source.metadata, ForkOptions(id="destination", scope="tree"), BACKGROUND_CONTEXT),
            return_exceptions=True,
        )
        strict_equal(isinstance(results[0], BaseException), False)
        strict_equal(isinstance(results[1], BaseException), True)
        if not isinstance(results[0], BaseException):
            await results[0].close(BACKGROUND_CONTEXT)
        await source.close(BACKGROUND_CONTEXT)

    async def fork_reserves_first(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        results = await asyncio.gather(
            repo.fork(source.metadata, ForkOptions(id="destination", scope="tree"), BACKGROUND_CONTEXT),
            repo.create(SessionCreateOptions(id="destination"), BACKGROUND_CONTEXT),
            return_exceptions=True,
        )
        strict_equal(isinstance(results[0], BaseException), False)
        strict_equal(isinstance(results[1], BaseException), True)
        if not isinstance(results[0], BaseException):
            await results[0].close(BACKGROUND_CONTEXT)
        await source.close(BACKGROUND_CONTEXT)

    return [
        _create_case(
            factory,
            "fork coordination",
            "publishes create when it reserves a shared destination id first",
            create_reserves_first,
        ),
        _create_case(
            factory,
            "fork coordination",
            "publishes fork when it reserves a shared destination id first",
            fork_reserves_first,
        ),
    ]


def create_session_repo_fork_source_snapshot_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create fork cases that require a snapshot boundary on an active source storage queue."""
    factory = _prepare_repo_case_factory(backend_factory, on_close)

    async def captures_coherent_boundary(context: _RepoCaseContext) -> None:
        repo = context.repo
        source = await repo.create(SessionCreateOptions(id="source"), BACKGROUND_CONTEXT)
        first_mutation = await source.begin_mutation(BACKGROUND_CONTEXT)
        first_commit = first_mutation.commit(
            [
                insert_entry(CustomEntry(id=ROOT_ID, parent_id=None, custom_type="first")),
                set_value(branch_tip("main"), ROOT_ID),
                set_value(lane_config("main"), _configuration()),
                set_value(lane_state("main"), _idle_lane_state()),
                set_value(session_name(), "first name"),
                set_value(entry_label(ROOT_ID), "first label"),
            ],
            BACKGROUND_CONTEXT,
        )
        fork_task = asyncio.ensure_future(
            repo.fork(
                source.metadata,
                ForkOptions(id="fork", scope="branch", branch="main"),
                BACKGROUND_CONTEXT,
            )
        )
        second_commit = asyncio.ensure_future(
            source.mutate(
                _create_repo_commit(
                    [
                        insert_entry(CustomEntry(id=CHILD_ID, parent_id=ROOT_ID, custom_type="second")),
                        set_value(branch_tip("main"), CHILD_ID),
                        set_value(session_name(), "second name"),
                        set_value(entry_label(ROOT_ID), "second label"),
                    ]
                ),
                BACKGROUND_CONTEXT,
            )
        )

        _, forked = await asyncio.gather(first_commit, fork_task)
        await first_mutation.end(BACKGROUND_CONTEXT)
        await second_commit
        strict_equal(await _get_branch_tip(forked), ROOT_ID)
        deep_strict_equal(
            [entry.id for entry in await forked.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)],
            [ROOT_ID],
        )
        strict_equal(await forked.get_name(BACKGROUND_CONTEXT), "first name")
        strict_equal(await forked.get_label(ROOT_ID, BACKGROUND_CONTEXT), "first label")
        await asyncio.gather(source.close(BACKGROUND_CONTEXT), forked.close(BACKGROUND_CONTEXT))

    return [
        _create_case(
            factory,
            "fork coordination",
            "captures one coherent boundary between source commits",
            captures_coherent_boundary,
        )
    ]


def create_session_repo_fork_coordination_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create every fork coordination case."""
    return [
        *create_session_repo_fork_destination_reservation_conformance(backend_factory, on_close),
        *create_session_repo_fork_source_snapshot_conformance(backend_factory, on_close),
    ]


def create_session_repo_fork_conformance(
    backend_factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create every fork conformance case."""
    return [
        *create_session_repo_fork_behavior_conformance(backend_factory, on_close),
        *create_session_repo_fork_coordination_conformance(backend_factory, on_close),
    ]


def create_session_repo_conformance(
    factory: Callable[[], Awaitable[SessionRepo]],
    on_close: Optional[Callable[[], Any]] = None,
) -> List[ConformanceCase]:
    """Create every SessionRepo conformance case."""
    return [
        *create_session_repo_lifecycle_conformance(factory, on_close),
        *create_session_repo_ownership_conformance(factory, on_close),
        *create_session_repo_message_conformance(factory, on_close),
        *create_session_repo_fork_conformance(factory, on_close),
    ]
