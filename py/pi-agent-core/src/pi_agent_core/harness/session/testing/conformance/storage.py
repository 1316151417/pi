"""Storage conformance cases ported from ``session/testing/conformance/storage.ts``.

The TypeScript table is a list of runner-independent cases created by
:func:`create_storage_conformance`. Every case is reproduced here with its
group, name, and assertions; ``node:assert/strict`` calls map onto the helpers
in :mod:`.assertions`.

Concurrency note: the TypeScript table relies on promises starting eagerly when
callers invoke ``storage.commit(...)``. Python coroutines do not start until they
are awaited, so the cases that need overlapping operations schedule them with
``asyncio.ensure_future`` (the counterpart of an already-started promise) and
await them with ``asyncio.gather`` (the counterpart of ``Promise.all``).
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Awaitable, Callable, List, Optional

from pi_ai.types import Cost, TextContent, Usage, UserMessage
from ....context import BACKGROUND_CONTEXT
from ...commit import insert_entry, insert_usage, materialize_committed_entry
from ...types import (
    CompactionEntry,
    CustomEntry,
    EntryCursor,
    EntryScan,
    EntryStructure,
    MessageEntry,
    SessionStats,
    Storage,
    StorageBranchScan,
    UsageRow,
    UsageScan,
)
from ...values import (
    ListCursor,
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    append_list,
    branch_tip,
    delete_list,
    delete_value,
    entry_label,
    list as list_address,
    pending_entry,
    set_value,
    value,
)
from ..types import ConformanceCase, StorageFixture
from .assertions import deep_strict_equal, ok, rejects, strict_equal, strictly_increasing

__all__ = ["create_storage_conformance"]

MESSAGE_TIMESTAMP = 1_650_000_000_000
_TEST_NAME = value("test.session.name")


def _test_value(key: str = "") -> Value:
    return value("test.value", key)


def _test_list(key: str = "") -> ValueList:
    return list_address("test.list", key)


def _usage(
    input: int,
    output: int,
    *,
    cache_write_1h: Optional[int] = None,
    reasoning: Optional[int] = None,
) -> Usage:
    """Build the exact usage row the TypeScript table compares against."""
    return Usage(
        input=input,
        output=output,
        cache_read=input + 1,
        cache_write=output + 1,
        cache_write_1h=cache_write_1h,
        reasoning=reasoning,
        total_tokens=input + output,
        cost=Cost(
            input=input / 100,
            output=output / 100,
            cache_read=(input + 1) / 100,
            cache_write=(output + 1) / 100,
            total=(input + output + 2) / 100,
        ),
    )


def _zero_usage() -> Usage:
    """Counterpart of ``zeroUsage()``: every counter and cost at zero."""
    return Usage(cost=Cost())


def _user_entry(entry_id: str, parent_id: Optional[str] = None, text: Optional[str] = None) -> MessageEntry:
    return MessageEntry(
        id=entry_id,
        parent_id=parent_id,
        message=UserMessage(
            content=[TextContent(text=entry_id if text is None else text)],
            timestamp=MESSAGE_TIMESTAMP,
        ),
    )


def _custom_entry(
    entry_id: str,
    parent_id: Optional[str],
    custom_type: str = "note",
    data: Optional[Any] = None,
) -> CustomEntry:
    return CustomEntry(
        id=entry_id,
        parent_id=parent_id,
        custom_type=custom_type,
        data={"id": entry_id} if data is None else data,
    )


def _compaction_entry(entry_id: str, parent_id: Optional[str]) -> CompactionEntry:
    return CompactionEntry(
        id=entry_id,
        parent_id=parent_id,
        summary=f"summary:{entry_id}",
        retained_tail=[],
        tokens_before=10,
        from_hook=False,
    )


def _ids(entries: List[Any]) -> List[str]:
    """EntryStructure is included for scanBranchStructure conformance."""
    return [entry.id for entry in entries]


def _usage_row(
    row_id: str,
    input: int,
    output: int,
    adjustment: bool,
    entry_id: Optional[str] = None,
) -> UsageRow:
    return UsageRow(id=row_id, usage=_usage(input, output), adjustment=adjustment, entry_id=entry_id)


def _stored(address: Any, stored_value: Any, seq: int) -> StoredValue:
    return StoredValue(address=address, value=stored_value, seq=seq)


async def _assert_commit_stats(storage: Storage, result: Any) -> None:
    deep_strict_equal(result.stats, await storage.get_stats(BACKGROUND_CONTEXT))


def _create_case(
    factory: Callable[[], Awaitable[StorageFixture]],
    group: str,
    name: str,
    test: Callable[[StorageFixture], Awaitable[None]],
) -> ConformanceCase:
    async def run() -> None:
        fixture = await factory()
        try:
            await test(fixture)
        finally:
            await fixture.aclose()

    return ConformanceCase(group=group, name=name, run=run)


def create_storage_conformance(
    factory: Callable[[], Awaitable[StorageFixture]],
) -> List[ConformanceCase]:
    """Create fresh, runner-independent cases for the durable Storage contract."""

    async def commits_mixed_writes(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(_user_entry("entry")),
                set_value(_TEST_NAME, "session"),
                insert_usage(_usage_row("usage", 2, 3, False, entry_id="entry")),
            ],
            BACKGROUND_CONTEXT,
        )

        strict_equal(len(result.seqs), 3)
        strict_equal(result.first_seq, result.seqs[0])
        await _assert_commit_stats(storage, result)
        strictly_increasing(result.seqs)
        ok(isinstance(result.timestamp, int) and result.timestamp >= 0)
        deep_strict_equal(
            await storage.get_entries(["entry"], BACKGROUND_CONTEXT),
            {"entry": materialize_committed_entry(_user_entry("entry"), result.seqs[0], result.timestamp)},
        )
        deep_strict_equal(
            await storage.get_value(_TEST_NAME, BACKGROUND_CONTEXT),
            _stored(_TEST_NAME, "session", result.seqs[1]),
        )
        deep_strict_equal(
            await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT),
            [
                UsageRow(
                    id="usage",
                    seq=result.seqs[2],
                    usage=_usage(2, 3),
                    adjustment=False,
                    entry_id="entry",
                )
            ],
        )

    async def rolls_back_mixed_transaction(fixture: StorageFixture) -> None:
        storage = fixture.storage
        await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_usage(_usage_row("taken", 1, 1, False)),
            ],
            BACKGROUND_CONTEXT,
        )
        entries_before = await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)
        usage_before = await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT)
        stats_before = await storage.get_stats(BACKGROUND_CONTEXT)

        await rejects(
            storage.commit(
                [
                    set_value(_TEST_NAME, "transient"),
                    insert_entry(_custom_entry("transient-entry", "root")),
                    insert_usage(_usage_row("transient-usage", 5, 8, True)),
                    insert_entry(_custom_entry("taken", "root")),
                ],
                BACKGROUND_CONTEXT,
            )
        )

        deep_strict_equal(
            await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT), entries_before
        )
        deep_strict_equal(
            await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT), usage_before
        )
        deep_strict_equal(await storage.get_stats(BACKGROUND_CONTEXT), stats_before)
        strict_equal(await storage.get_value(_TEST_NAME, BACKGROUND_CONTEXT), None)

    async def preserves_overwritten_and_deleted(fixture: StorageFixture) -> None:
        storage = fixture.storage
        await storage.commit(
            [
                set_value(_test_value("overwritten"), "original"),
                set_value(_test_value("deleted"), {"kept": True}),
                insert_entry(_user_entry("taken")),
            ],
            BACKGROUND_CONTEXT,
        )
        overwritten_before = await storage.get_value(_test_value("overwritten"), BACKGROUND_CONTEXT)
        deleted_before = await storage.get_value(_test_value("deleted"), BACKGROUND_CONTEXT)

        await rejects(
            storage.commit(
                [
                    set_value(_test_value("overwritten"), "transient"),
                    delete_value(_test_value("deleted")),
                    insert_entry(_custom_entry("transient", "taken")),
                    insert_entry(_custom_entry("taken", None)),
                ],
                BACKGROUND_CONTEXT,
            )
        )

        deep_strict_equal(
            await storage.get_value(_test_value("overwritten"), BACKGROUND_CONTEXT),
            overwritten_before,
        )
        deep_strict_equal(
            await storage.get_value(_test_value("deleted"), BACKGROUND_CONTEXT), deleted_before
        )
        strict_equal("transient" in await storage.get_entries(["transient"], BACKGROUND_CONTEXT), False)

    async def enforces_shared_id_namespace(fixture: StorageFixture) -> None:
        storage = fixture.storage
        await storage.commit(
            [
                insert_entry(_user_entry("existing-entry")),
                insert_usage(_usage_row("existing-usage", 1, 1, False)),
            ],
            BACKGROUND_CONTEXT,
        )

        await rejects(
            storage.commit(
                [insert_usage(_usage_row("existing-entry", 2, 2, False))], BACKGROUND_CONTEXT
            )
        )
        await rejects(
            storage.commit([insert_entry(_custom_entry("existing-usage", None))], BACKGROUND_CONTEXT)
        )

        duplicates: List[tuple] = [
            (
                "entry-then-usage",
                [
                    insert_entry(_custom_entry("entry-then-usage", None)),
                    insert_usage(_usage_row("entry-then-usage", 3, 3, False)),
                ],
            ),
            (
                "usage-then-entry",
                [
                    insert_usage(_usage_row("usage-then-entry", 4, 4, False)),
                    insert_entry(_custom_entry("usage-then-entry", None)),
                ],
            ),
        ]
        for entry_id, writes in duplicates:
            await rejects(
                storage.commit(writes, BACKGROUND_CONTEXT),
                f"Expected duplicate id {entry_id} to reject",
            )

        deep_strict_equal(
            _ids(await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)),
            ["existing-entry"],
        )
        deep_strict_equal(
            [row.id for row in await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT)],
            ["existing-usage"],
        )

    async def resolves_parents_in_write_order(fixture: StorageFixture) -> None:
        storage = fixture.storage
        await storage.commit([insert_entry(_user_entry("root"))], BACKGROUND_CONTEXT)
        await storage.commit(
            [
                insert_entry(_custom_entry("child", "root")),
                insert_entry(_custom_entry("grandchild", "child")),
            ],
            BACKGROUND_CONTEXT,
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(start="grandchild", order="oldestFirst"), BACKGROUND_CONTEXT
                )
            ),
            ["root", "child", "grandchild"],
        )

        await rejects(
            storage.commit(
                [
                    insert_entry(_custom_entry("before-parent", "later-parent")),
                    insert_entry(_custom_entry("later-parent", "root")),
                    set_value(entry_label("before-parent"), "transient"),
                ],
                BACKGROUND_CONTEXT,
            )
        )
        await rejects(
            storage.commit([insert_entry(_custom_entry("orphan", "missing"))], BACKGROUND_CONTEXT)
        )
        await storage.commit(
            [insert_usage(_usage_row("usage-is-not-parent", 1, 1, False))], BACKGROUND_CONTEXT
        )
        await rejects(
            storage.commit(
                [insert_entry(_custom_entry("usage-child", "usage-is-not-parent"))], BACKGROUND_CONTEXT
            )
        )

        deep_strict_equal(
            await storage.get_entries(
                ["before-parent", "later-parent", "orphan", "usage-child"], BACKGROUND_CONTEXT
            ),
            {},
        )
        strict_equal(await storage.get_value(entry_label("before-parent"), BACKGROUND_CONTEXT), None)

    async def places_pending_content(fixture: StorageFixture) -> None:
        storage = fixture.storage
        entry = _user_entry("reserved", None, "queued")
        await storage.commit(
            [
                set_value(pending_entry(entry.id), {"type": "message", "payload": entry.message}),
                set_value(branch_tip("main"), None),
            ],
            BACKGROUND_CONTEXT,
        )

        strict_equal(entry.id in await storage.get_entries([entry.id], BACKGROUND_CONTEXT), False)
        stored_pending = await storage.get_value(pending_entry(entry.id), BACKGROUND_CONTEXT)
        deep_strict_equal(
            None if stored_pending is None else stored_pending.value,
            {"type": "message", "payload": entry.message},
        )
        stored_tip = await storage.get_value(branch_tip("main"), BACKGROUND_CONTEXT)
        strict_equal(None if stored_tip is None else stored_tip.value, None)

        placement = await storage.commit(
            [
                insert_entry(entry),
                delete_value(pending_entry(entry.id)),
                set_value(branch_tip("main"), entry.id),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            await storage.get_entries([entry.id], BACKGROUND_CONTEXT),
            {entry.id: materialize_committed_entry(entry, placement.seqs[0], placement.timestamp)},
        )
        strict_equal(await storage.get_value(pending_entry(entry.id), BACKGROUND_CONTEXT), None)
        deep_strict_equal(
            await storage.get_value(branch_tip("main"), BACKGROUND_CONTEXT),
            _stored(branch_tip("main"), entry.id, placement.seqs[2]),
        )

    async def sets_replaces_deletes_values(fixture: StorageFixture) -> None:
        storage = fixture.storage
        first = await storage.commit(
            [
                set_value(_test_value("prefix/b"), 1),
                set_value(_test_value("prefix/a"), 2),
                set_value(_test_value("other"), 3),
                set_value(_test_value("prefix/\ue000"), 4),
                set_value(_test_value("prefix/\U00010000"), 5),
                set_value(_test_value("prefix/a"), None),
            ],
            BACKGROUND_CONTEXT,
        )
        deep_strict_equal(
            await storage.get_value(_test_value("prefix/a"), BACKGROUND_CONTEXT),
            _stored(_test_value("prefix/a"), None, first.seqs[5]),
        )

        second = await storage.commit(
            [
                delete_value(_test_value("prefix/a")),
                delete_value(_test_value("absent")),
                set_value(_test_value("prefix/a"), "recreated"),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            await storage.scan_values(_test_value("prefix/"), BACKGROUND_CONTEXT),
            [
                _stored(_test_value("prefix/a"), "recreated", second.seqs[2]),
                _stored(_test_value("prefix/b"), 1, first.seqs[0]),
                _stored(_test_value("prefix/\ue000"), 4, first.seqs[3]),
                _stored(_test_value("prefix/\U00010000"), 5, first.seqs[4]),
            ],
        )
        strict_equal(await storage.get_value(_test_value("absent"), BACKGROUND_CONTEXT), None)

    async def applies_value_and_list_write_order(fixture: StorageFixture) -> None:
        storage = fixture.storage
        kept_value = _test_value("write-order/kept")
        deleted_value = _test_value("write-order/deleted")
        kept_list = _test_list("write-order/kept")
        deleted_list = _test_list("write-order/deleted")
        result = await storage.commit(
            [
                set_value(deleted_value, "transient"),
                delete_value(deleted_value),
                set_value(kept_value, "transient"),
                set_value(kept_value, "kept"),
                append_list(kept_list, "transient"),
                delete_list(kept_list),
                append_list(kept_list, "kept"),
                append_list(deleted_list, "transient"),
                delete_list(deleted_list),
            ],
            BACKGROUND_CONTEXT,
        )

        strict_equal(await storage.get_value(deleted_value, BACKGROUND_CONTEXT), None)
        deep_strict_equal(
            await storage.get_value(kept_value, BACKGROUND_CONTEXT),
            _stored(kept_value, "kept", result.seqs[3]),
        )
        deep_strict_equal(
            await storage.read_list(kept_list, None, BACKGROUND_CONTEXT),
            [ListElement(seq=result.seqs[6], value="kept")],
        )
        deep_strict_equal(await storage.read_list(deleted_list, None, BACKGROUND_CONTEXT), [])

    async def keeps_history_during_value_commits(fixture: StorageFixture) -> None:
        storage = fixture.storage
        await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_usage(_usage_row("historical-usage", 2, 3, False)),
            ],
            BACKGROUND_CONTEXT,
        )
        entries_before = await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)
        usage_before = await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT)
        stats_before = await storage.get_stats(BACKGROUND_CONTEXT)

        result = await storage.commit(
            [set_value(_TEST_NAME, "first"), set_value(_TEST_NAME, "second")],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT), entries_before
        )
        deep_strict_equal(
            await storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT), usage_before
        )
        deep_strict_equal(await storage.get_stats(BACKGROUND_CONTEXT), stats_before)
        deep_strict_equal(
            await storage.get_value(_TEST_NAME, BACKGROUND_CONTEXT),
            _stored(_TEST_NAME, "second", result.seqs[1]),
        )

    async def pages_appends_and_deletes_lists(fixture: StorageFixture) -> None:
        storage = fixture.storage
        address = _test_list("events")
        strict_equal(len(await storage.read_list(address, None, BACKGROUND_CONTEXT)), 0)
        result = await storage.commit(
            [
                append_list(address, "a"),
                set_value(_TEST_NAME, "gap"),
                append_list(address, "b"),
                append_list(address, "c"),
            ],
            BACKGROUND_CONTEXT,
        )
        deep_strict_equal(
            await storage.read_list(address, None, BACKGROUND_CONTEXT),
            [
                ListElement(seq=result.seqs[0], value="a"),
                ListElement(seq=result.seqs[2], value="b"),
                ListElement(seq=result.seqs[3], value="c"),
            ],
        )
        deep_strict_equal(
            await storage.read_list(address, ListReadOptions(limit=2), BACKGROUND_CONTEXT),
            [
                ListElement(seq=result.seqs[0], value="a"),
                ListElement(seq=result.seqs[2], value="b"),
            ],
        )
        deep_strict_equal(
            await storage.read_list(
                address,
                ListReadOptions(cursor=ListCursor(seq=result.seqs[0]), limit=2),
                BACKGROUND_CONTEXT,
            ),
            [
                ListElement(seq=result.seqs[2], value="b"),
                ListElement(seq=result.seqs[3], value="c"),
            ],
        )
        deep_strict_equal(
            await storage.read_list(
                address, ListReadOptions(order="desc", limit=2), BACKGROUND_CONTEXT
            ),
            [
                ListElement(seq=result.seqs[3], value="c"),
                ListElement(seq=result.seqs[2], value="b"),
            ],
        )
        deep_strict_equal(
            await storage.read_list(
                address,
                ListReadOptions(order="desc", cursor=ListCursor(seq=result.seqs[3]), limit=2),
                BACKGROUND_CONTEXT,
            ),
            [
                ListElement(seq=result.seqs[2], value="b"),
                ListElement(seq=result.seqs[0], value="a"),
            ],
        )
        await rejects(storage.read_list(address, ListReadOptions(limit=0), BACKGROUND_CONTEXT))
        # ``Number.MAX_VALUE`` is not a safe integer; the Python counterpart is a
        # float limit, which is not a positive integer either.
        await rejects(
            storage.read_list(
                address, ListReadOptions(limit=sys.float_info.max), BACKGROUND_CONTEXT
            )
        )

        await storage.commit(
            [delete_list(address), delete_list(_test_list("absent")), append_list(address, "new")],
            BACKGROUND_CONTEXT,
        )
        deep_strict_equal(
            [
                element.value
                for element in await storage.read_list(address, None, BACKGROUND_CONTEXT)
            ],
            ["new"],
        )

    async def clamps_one_read_page(fixture: StorageFixture) -> None:
        storage = fixture.storage
        address = _test_list("large")
        await storage.commit(
            [append_list(address, index) for index in range(10_001)],
            BACKGROUND_CONTEXT,
        )
        first_page = await storage.read_list(address, None, BACKGROUND_CONTEXT)
        strict_equal(len(first_page), 1_000)
        strict_equal(
            len(await storage.read_list(address, ListReadOptions(limit=20_000), BACKGROUND_CONTEXT)),
            10_000,
        )
        strict_equal(
            len(
                await storage.read_list(
                    address,
                    ListReadOptions(cursor=ListCursor(seq=first_page[-1].seq)),
                    BACKGROUND_CONTEXT,
                )
            ),
            1_000,
        )

    async def commits_mixed_list_writes(fixture: StorageFixture) -> None:
        storage = fixture.storage
        address = _test_list("atomic")
        committed = await storage.commit(
            [
                insert_entry(_user_entry("mixed")),
                append_list(address, "kept"),
                set_value(_TEST_NAME, "kept"),
                insert_usage(_usage_row("mixed-usage", 1, 2, False)),
            ],
            BACKGROUND_CONTEXT,
        )
        deep_strict_equal(
            await storage.read_list(address, None, BACKGROUND_CONTEXT),
            [ListElement(seq=committed.seqs[1], value="kept")],
        )

        await rejects(
            storage.commit(
                [
                    append_list(address, "transient"),
                    delete_value(_TEST_NAME),
                    insert_entry(_user_entry("mixed")),
                ],
                BACKGROUND_CONTEXT,
            )
        )
        deep_strict_equal(
            await storage.read_list(address, None, BACKGROUND_CONTEXT),
            [ListElement(seq=committed.seqs[1], value="kept")],
        )
        stored = await storage.get_value(_TEST_NAME, BACKGROUND_CONTEXT)
        strict_equal(None if stored is None else stored.value, "kept")

    async def stores_custom_entries(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(CustomEntry(id="without-data", parent_id=None, custom_type="marker")),
                insert_entry(_custom_entry("with-data", "without-data", "note", {"nested": [1, 2]})),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            await storage.get_entries(["without-data", "with-data"], BACKGROUND_CONTEXT),
            {
                "without-data": CustomEntry(
                    id="without-data",
                    parent_id=None,
                    custom_type="marker",
                    seq=result.seqs[0],
                    timestamp=result.timestamp,
                ),
                "with-data": materialize_committed_entry(
                    _custom_entry("with-data", "without-data", "note", {"nested": [1, 2]}),
                    result.seqs[1],
                    result.timestamp,
                ),
            },
        )

    async def scans_global_entries(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_entry(_custom_entry("note-1", "root", "note")),
                insert_entry(_custom_entry("other", "note-1", "other")),
                insert_entry(_custom_entry("note-2", "other", "note")),
                insert_entry(_user_entry("tail", "note-2")),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            _ids(
                await storage.scan_entries(
                    EntryScan(
                        type="custom",
                        custom_type="note",
                        from_seq=result.seqs[1],
                        to_seq=result.seqs[3],
                        order="desc",
                    ),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["note-2", "note-1"],
        )
        deep_strict_equal(
            _ids(await storage.scan_entries(EntryScan(order="asc", limit=2), BACKGROUND_CONTEXT)),
            ["root", "note-1"],
        )
        deep_strict_equal(
            _ids(await storage.scan_entries(EntryScan(order="desc", limit=2), BACKGROUND_CONTEXT)),
            ["tail", "note-2"],
        )

    async def applies_stops_before_filters(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_entry(_custom_entry("marker", "root", "marker")),
                insert_entry(_user_entry("middle", "marker")),
                insert_entry(_compaction_entry("compact", "middle")),
                insert_entry(_custom_entry("note", "compact", "note")),
                insert_entry(_user_entry("leaf", "note")),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(start="leaf", stop_at_type="compaction", type="message"),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["leaf"],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(
                        start="leaf", order="oldestFirst", stop_at_id="middle", type="custom"
                    ),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["marker"],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(
                        start="leaf",
                        order="newestFirst",
                        cursor=EntryCursor(seq=result.seqs[4]),
                        limit=2,
                    ),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["compact", "middle"],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(
                        start="leaf",
                        order="oldestFirst",
                        cursor=EntryCursor(seq=result.seqs[1]),
                        limit=2,
                    ),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["middle", "compact"],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(start="leaf", stop_at_id="leaf", type="custom"), BACKGROUND_CONTEXT
                )
            ),
            [],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch(
                    StorageBranchScan(start="leaf", custom_type="note"), BACKGROUND_CONTEXT
                )
            ),
            ["note"],
        )
        await rejects(storage.scan_branch(StorageBranchScan(start="missing"), BACKGROUND_CONTEXT))

    async def returns_branch_structure(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_entry(_custom_entry("child", "root", "note")),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            await storage.scan_branch_structure(
                StorageBranchScan(start="child", order="oldestFirst"), BACKGROUND_CONTEXT
            ),
            [
                EntryStructure(
                    id="root",
                    parent_id=None,
                    seq=result.seqs[0],
                    timestamp=result.timestamp,
                    type="message",
                ),
                EntryStructure(
                    id="child",
                    parent_id="root",
                    seq=result.seqs[1],
                    timestamp=result.timestamp,
                    type="custom",
                    custom_type="note",
                ),
            ],
        )

    async def applies_branch_semantics_to_structure(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_entry(_user_entry("root")),
                insert_entry(_custom_entry("marker", "root", "marker")),
                insert_entry(_user_entry("middle", "marker")),
                insert_entry(_compaction_entry("compact", "middle")),
                insert_entry(_custom_entry("note", "compact", "note")),
                insert_entry(_user_entry("leaf", "note")),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            _ids(
                await storage.scan_branch_structure(
                    StorageBranchScan(start="leaf", stop_at_type="compaction", type="message"),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["leaf"],
        )
        deep_strict_equal(
            _ids(
                await storage.scan_branch_structure(
                    StorageBranchScan(
                        start="leaf",
                        order="oldestFirst",
                        cursor=EntryCursor(seq=result.seqs[1]),
                        limit=2,
                    ),
                    BACKGROUND_CONTEXT,
                )
            ),
            ["middle", "compact"],
        )
        await rejects(
            storage.scan_branch_structure(StorageBranchScan(start="missing"), BACKGROUND_CONTEXT)
        )

    async def scans_usage_ledger(fixture: StorageFixture) -> None:
        storage = fixture.storage
        result = await storage.commit(
            [
                insert_usage(_usage_row("usage-1", 1, 1, False)),
                set_value(_TEST_NAME, "sequence gap"),
                insert_usage(_usage_row("usage-2", 2, 2, False)),
                insert_usage(_usage_row("usage-3", 3, 3, True)),
            ],
            BACKGROUND_CONTEXT,
        )

        deep_strict_equal(
            [
                row.id
                for row in await storage.scan_usage(
                    UsageScan(from_seq=result.seqs[1], to_seq=result.seqs[2], order="asc"),
                    BACKGROUND_CONTEXT,
                )
            ],
            ["usage-2"],
        )
        deep_strict_equal(
            [
                row.id
                for row in await storage.scan_usage(UsageScan(order="desc", limit=2), BACKGROUND_CONTEXT)
            ],
            ["usage-3", "usage-2"],
        )
        deep_strict_equal(
            [
                row.id
                for row in await storage.scan_usage(UsageScan(order="asc", limit=2), BACKGROUND_CONTEXT)
            ],
            ["usage-1", "usage-2"],
        )

    async def keeps_stats_equal_to_ledger(fixture: StorageFixture) -> None:
        storage = fixture.storage
        deep_strict_equal(
            await storage.get_stats(BACKGROUND_CONTEXT),
            SessionStats(message_count=0, usage=_zero_usage()),
        )

        first_usage = _usage(2, 3, cache_write_1h=4, reasoning=1)
        first = await storage.commit(
            [
                insert_entry(_user_entry("message")),
                insert_usage(UsageRow(id="usage-1", usage=first_usage, adjustment=False)),
            ],
            BACKGROUND_CONTEXT,
        )
        await _assert_commit_stats(storage, first)
        deep_strict_equal(first.stats, SessionStats(message_count=1, usage=first_usage))

        second_usage = _usage(5, 7, cache_write_1h=6, reasoning=2)
        second = await storage.commit(
            [
                insert_entry(_custom_entry("custom", "message")),
                insert_entry(_compaction_entry("compaction", "custom")),
                insert_usage(UsageRow(id="usage-2", usage=second_usage, adjustment=True)),
            ],
            BACKGROUND_CONTEXT,
        )
        await _assert_commit_stats(storage, second)
        deep_strict_equal(
            second.stats,
            SessionStats(
                message_count=1,
                usage=Usage(
                    input=7,
                    output=10,
                    cache_read=9,
                    cache_write=12,
                    cache_write_1h=10,
                    reasoning=3,
                    total_tokens=17,
                    cost=Cost(
                        input=first_usage.cost.input + second_usage.cost.input,
                        output=first_usage.cost.output + second_usage.cost.output,
                        cache_read=first_usage.cost.cache_read + second_usage.cost.cache_read,
                        cache_write=first_usage.cost.cache_write + second_usage.cost.cache_write,
                        total=first_usage.cost.total + second_usage.cost.total,
                    ),
                ),
            ),
        )

    async def serializes_back_to_back_commits(fixture: StorageFixture) -> None:
        storage = fixture.storage
        # ``ensure_future`` mirrors the eager start of a TypeScript promise: the
        # first commit is admitted before the second one is created.
        first = asyncio.ensure_future(
            storage.commit([insert_entry(_user_entry("first"))], BACKGROUND_CONTEXT)
        )
        second = asyncio.ensure_future(
            storage.commit([insert_entry(_user_entry("second", "first"))], BACKGROUND_CONTEXT)
        )
        first_result, second_result = await asyncio.gather(first, second)

        ok(first_result.seqs[0] < second_result.seqs[0])
        deep_strict_equal(
            first_result.stats, SessionStats(message_count=1, usage=_zero_usage())
        )
        deep_strict_equal(
            second_result.stats, SessionStats(message_count=2, usage=_zero_usage())
        )
        await _assert_commit_stats(storage, second_result)
        deep_strict_equal(
            _ids(await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)),
            ["first", "second"],
        )

    async def seals_admission_and_closes(fixture: StorageFixture) -> None:
        storage = fixture.storage
        admitted = asyncio.ensure_future(
            storage.commit([insert_entry(_user_entry("admitted"))], BACKGROUND_CONTEXT)
        )
        first_close = asyncio.ensure_future(storage.close(BACKGROUND_CONTEXT))
        second_close = asyncio.ensure_future(storage.close(BACKGROUND_CONTEXT))
        await asyncio.sleep(0)

        await rejects(storage.get_stats(BACKGROUND_CONTEXT))
        await rejects(storage.commit([], BACKGROUND_CONTEXT))
        strict_equal(len((await admitted).seqs), 1)
        await asyncio.gather(first_close, second_close)

        rejected_reads = [
            storage.get_entries([], BACKGROUND_CONTEXT),
            storage.get_value(_TEST_NAME, BACKGROUND_CONTEXT),
            storage.scan_values(_TEST_NAME, BACKGROUND_CONTEXT),
            storage.read_list(_test_list("events"), None, BACKGROUND_CONTEXT),
            storage.scan_branch(StorageBranchScan(start="admitted"), BACKGROUND_CONTEXT),
            storage.scan_branch_structure(StorageBranchScan(start="admitted"), BACKGROUND_CONTEXT),
            storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT),
            storage.scan_usage(UsageScan(order="asc"), BACKGROUND_CONTEXT),
            storage.get_stats(BACKGROUND_CONTEXT),
        ]
        for read in rejected_reads:
            await rejects(read)

    return [
        _create_case(
            factory,
            "transactions",
            "commits mixed writes atomically in write order",
            commits_mixed_writes,
        ),
        _create_case(
            factory,
            "transactions",
            "rolls back every store when a mixed transaction fails",
            rolls_back_mixed_transaction,
        ),
        _create_case(
            factory,
            "transactions",
            "preserves overwritten and deleted values when a transaction fails",
            preserves_overwritten_and_deleted,
        ),
        _create_case(
            factory,
            "transactions",
            "enforces one shared entry and usage id namespace",
            enforces_shared_id_namespace,
        ),
        _create_case(
            factory,
            "transactions",
            "resolves parents only from prior entries and earlier writes",
            resolves_parents_in_write_order,
        ),
        _create_case(
            factory,
            "transactions",
            "places pending content under its reserved entry id",
            places_pending_content,
        ),
        _create_case(
            factory,
            "values",
            "sets, replaces, deletes, and recreates values without tombstones",
            sets_replaces_deletes_values,
        ),
        _create_case(
            factory,
            "values",
            "applies same-transaction value and list operations in write order",
            applies_value_and_list_write_order,
        ),
        _create_case(
            factory,
            "values",
            "does not change historical stores during value-only commits",
            keeps_history_during_value_commits,
        ),
        _create_case(
            factory,
            "lists",
            "pages appends by global sequence and deletes whole lists",
            pages_appends_and_deletes_lists,
        ),
        _create_case(
            factory,
            "lists",
            "clamps one read page without limiting list growth",
            clamps_one_read_page,
        ),
        _create_case(
            factory,
            "lists",
            "commits mixed list writes atomically and rolls them back with siblings",
            commits_mixed_list_writes,
        ),
        _create_case(
            factory,
            "entry queries",
            "stores custom entries with and without data",
            stores_custom_entries,
        ),
        _create_case(
            factory,
            "entry queries",
            "scans global entries with explicit ranges, filters, orders, and limits",
            scans_global_entries,
        ),
        _create_case(
            factory,
            "branch queries",
            "applies stops before filters and cursors before limits",
            applies_stops_before_filters,
        ),
        _create_case(
            factory,
            "branch queries",
            "returns branch structure without payload fields",
            returns_branch_structure,
        ),
        _create_case(
            factory,
            "branch queries",
            "applies branch query semantics to structure scans",
            applies_branch_semantics_to_structure,
        ),
        _create_case(
            factory,
            "usage and stats",
            "scans the usage ledger with explicit ranges, orders, and limits",
            scans_usage_ledger,
        ),
        _create_case(
            factory,
            "usage and stats",
            "keeps stats equal to message count and ledger totals",
            keeps_stats_equal_to_ledger,
        ),
        _create_case(
            factory,
            "serialization",
            "serializes back-to-back commits in admission order",
            serializes_back_to_back_commits,
        ),
        _create_case(
            factory,
            "lifecycle",
            "seals admission, drains admitted commits, and closes idempotently",
            seals_admission_and_closes,
        ),
    ]
