"""JSONL storage ported from ``harness/pico3/jsonl.ts``.

Files in a session directory:

* ``main.jsonl`` — conversations, entries, inputs, rewindable/session doc ops,
  each task's CREATE and TERMINAL records, and one marker per commit.
  Append-only, never rewritten.
* ``sticky-<conversation>.jsonl`` — sticky doc ops; rewritten from its last base
  by ``truncate`` (on the session line).
* ``task-<id>.jsonl`` — a live task's intermediate patches (running,
  checkpoints); unlinked after its terminal record is published in main.

Publication (§10): one commit ``Seq``; sidecar records are appended first, then
exactly one main record, last, listing the sidecar refs it expects — the
publication point. Replay applies a sidecar record only when main has the marker
for that seq naming that file; unconfirmed sidecar tails are ignored. Every
record carries the committed-ID high-water. Bytes after the last newline of any
file are a torn write and are truncated before the file is opened for append.
There is no compaction.

One process owns a directory at a time; a second process is unsupported.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..._chord.context import BACKGROUND_CONTEXT, Context
from .delta import Op, is_base
from .memory import MemoryStorage
from .types import DocRef, Id, JsonObject, Seq, Storage, Write, write_from_json, write_to_json

__all__ = ["JsonlStorage"]


def _join(directory: str, name: str) -> str:
    return os.path.join(directory, name)


def _truncate_to(path: str, length: int) -> None:
    with open(path, "r+b") as handle:
        handle.truncate(length)


def _truncate_torn_tail(path: str) -> None:
    """Bytes after the last newline are a torn write: cut them."""
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as handle:
        data = handle.read()
    end = data.rfind(b"\n")
    if end + 1 != len(data):
        _truncate_to(path, end + 1)


class JsonlStorage(MemoryStorage):
    """Append-only JSONL storage with a memory read path."""

    def __init__(self, directory: str, fsync: bool = True) -> None:
        super().__init__()
        self.dir = directory
        self.fsync = fsync
        self._main = open(_join(directory, "main.jsonl"), "ab")
        self._sidecars: Dict[Id, Any] = {}
        self._task_sidecars: Dict[Id, Any] = {}
        self._closed_once = False

    @staticmethod
    async def open(directory: str, opts: Optional[Dict[str, Any]] = None) -> "JsonlStorage":
        opts = opts or {}
        os.makedirs(directory, exist_ok=True)
        for name in sorted(os.listdir(directory)):
            if name.endswith(".jsonl"):
                _truncate_torn_tail(_join(directory, name))
        storage = JsonlStorage(directory, bool(opts.get("fsync", True)))
        try:
            await storage.replay()
            return storage
        except BaseException:
            await storage.close()
            raise

    def sizes(self) -> Dict[str, int]:
        """For tests and tooling: sizes of every file in the directory."""
        return {
            name: os.path.getsize(_join(self.dir, name)) for name in sorted(os.listdir(self.dir))
        }

    # -- replay ------------------------------------------------------------

    async def replay(self) -> None:
        def read(file: str) -> List[Tuple[Dict[str, Any], int]]:
            if not os.path.exists(file):
                return []
            with open(file, "rb") as handle:
                data = handle.read()
            out: List[Tuple[Dict[str, Any], int]] = []
            start = 0
            last = 0
            while True:
                end = data.find(b"\n", start)
                if end < 0:
                    break
                line = data[start:end].decode("utf-8")
                start = end + 1
                if line == "":
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"{file}: malformed record: {error}") from error
                if (
                    not isinstance(record.get("seq"), int)
                    or not isinstance(record.get("maxId"), int)
                    or not isinstance(record.get("writes"), list)
                ):
                    raise ValueError(f"{file}: record lacks seq/maxId/writes")
                if record["seq"] <= last:
                    raise ValueError(f"{file}: sequence not increasing at {record['seq']}")
                last = record["seq"]
                out.append((record, start))
            return out

        max_id = 0
        main = [record for record, _end in read(_join(self.dir, "main.jsonl"))]
        confirmed: Dict[Seq, set] = {}
        for record in main:
            confirmed[record["seq"]] = set(record.get("refs") or [])

        sidecar_files = [
            name
            for name in sorted(os.listdir(self.dir))
            if name.startswith("sticky-") or name.startswith("task-")
        ]
        by_seq: Dict[Seq, List[Write]] = {}
        found: Dict[Seq, set] = {}
        retained_sticky_base: Dict[str, Seq] = {}
        retired_tasks: Dict[Id, Seq] = {}
        for record in main:
            for write in record["writes"]:
                if write.get("type") == "task.patch" and (write.get("patch") or {}).get(
                    "status"
                ) == "terminal":
                    retired_tasks[write["patch"]["id"]] = record["seq"]
            by_seq[record["seq"]] = [write_from_json(write) for write in record["writes"]]
            max_id = max(max_id, record["maxId"])

        for name in sidecar_files:
            path = _join(self.dir, name)
            records = read(path)
            first = records[0][0] if records else None
            if name.startswith("sticky-") and first is not None:
                ops: List[Op] = []
                for write in first.get("writes") or []:
                    if write.get("type") == "doc":
                        ops.extend(tuple(op) for op in write.get("ops") or [])
                if is_base(ops):
                    retained_sticky_base[name] = first["seq"]
            confirmed_end = 0
            saw_unconfirmed = False
            for record, end in records:
                if name not in confirmed.get(record["seq"], set()):
                    saw_unconfirmed = True
                    continue
                if saw_unconfirmed:
                    raise ValueError(
                        f"{path}: confirmed record follows an unconfirmed tail at {record['seq']}"
                    )
                confirmed_end = end
                files = found.setdefault(record["seq"], set())
                files.add(name)
                by_seq[record["seq"]].extend(write_from_json(write) for write in record["writes"])
                max_id = max(max_id, record["maxId"])
            if saw_unconfirmed:
                _truncate_to(path, confirmed_end)

        for record in main:
            for name in record.get("refs") or []:
                if name in found.get(record["seq"], set()):
                    continue
                retained_from = retained_sticky_base.get(name)
                if retained_from is not None and record["seq"] < retained_from:
                    continue
                if name.startswith("task-"):
                    id = int(name[5:-6])
                    retired_at = retired_tasks.get(id)
                    if retired_at is not None and record["seq"] < retired_at:
                        continue
                raise ValueError(
                    f"{_join(self.dir, name)}: missing record for published sequence {record['seq']}"
                )

        for seq in sorted(by_seq.keys()):
            self.seq = seq - 1
            await MemoryStorage.commit(self, by_seq[seq], BACKGROUND_CONTEXT)
        self.set_next_id(max_id + 1)

        # Task sidecars belong to live tasks only.
        for name in sidecar_files:
            if not name.startswith("task-"):
                continue
            id = int(name[5:-6])
            task = await MemoryStorage.task(self, id)
            if task is None or task.status == "terminal":
                try:
                    os.unlink(_join(self.dir, name))
                except FileNotFoundError:
                    pass

    # -- commit ------------------------------------------------------------

    async def commit(self, writes: List[Write], ctx: Optional[Context] = None) -> Seq:
        if self._closed_once:
            raise RuntimeError("storage closed")
        seq = self.seq + 1
        apply = self.stage(writes, seq)  # validates the whole batch first

        def next_id_after() -> Id:
            maximum = self.next_id_value - 1
            for write in writes:
                if write.type == "conversation":
                    id = write.conversation.id
                elif write.type == "entry":
                    id = write.entry.id
                elif write.type == "task":
                    id = write.task.id
                elif write.type == "input":
                    id = write.input.id
                else:
                    id = 0
                maximum = max(maximum, id)
            return maximum

        max_id = next_id_after()
        main: List[Write] = []
        sticky: Dict[Id, List[Write]] = {}
        tasks: Dict[Id, List[Write]] = {}
        retired: List[Id] = []
        for write in writes:
            if write.type == "doc" and write.ref.doc == "sticky":
                sticky.setdefault(write.ref.conversation_id, []).append(write)
            elif write.type == "task.patch" and write.patch.status != "terminal":
                tasks.setdefault(write.patch.id, []).append(write)
            else:
                main.append(write)
                if write.type == "task.patch" and write.patch.status == "terminal":
                    retired.append(write.patch.id)

        refs: List[str] = []
        for id, task_writes in tasks.items():
            self._append(self._task_sidecar(id), seq, max_id, task_writes)
            refs.append(f"task-{id}.jsonl")
        for conversation_id, sticky_writes in sticky.items():
            self._append(self._sidecar(conversation_id), seq, max_id, sticky_writes)
            refs.append(f"sticky-{conversation_id}.jsonl")
        # The publication point.
        self._append(self._main, seq, max_id, main, refs or None)
        apply()
        self.seq = seq
        for id in retired:
            self._retire_task_sidecar(id)
        return seq

    def _append(
        self,
        handle: Any,
        seq: Seq,
        max_id: Id,
        writes: List[Write],
        refs: Optional[List[str]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "seq": seq,
            "maxId": max_id,
            "writes": [write_to_json(write) for write in writes],
        }
        if refs:
            record["refs"] = refs
        handle.write((json.dumps(record) + "\n").encode("utf-8"))
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())

    def _sidecar(self, conversation_id: Id) -> Any:
        handle = self._sidecars.get(conversation_id)
        if handle is None:
            handle = open(_join(self.dir, f"sticky-{conversation_id}.jsonl"), "ab")
            self._sidecars[conversation_id] = handle
        return handle

    def _task_sidecar(self, id: Id) -> Any:
        handle = self._task_sidecars.get(id)
        if handle is None:
            handle = open(_join(self.dir, f"task-{id}.jsonl"), "ab")
            self._task_sidecars[id] = handle
        return handle

    def _retire_task_sidecar(self, id: Id) -> None:
        handle = self._task_sidecars.pop(id, None)
        if handle is not None:
            handle.close()
        try:
            os.unlink(_join(self.dir, f"task-{id}.jsonl"))
        except FileNotFoundError:
            pass

    # -- truncate ----------------------------------------------------------

    async def truncate(self, ref: DocRef, ctx: Optional[Context] = None) -> None:
        """Rewrite the sticky sidecar from its last base: temp file, optional fsync, rename."""
        await MemoryStorage.truncate(self, ref)
        if ref.doc != "sticky":
            return
        file = _join(self.dir, f"sticky-{ref.conversation_id}.jsonl")
        if not os.path.exists(file):
            return
        with open(file, "r", encoding="utf-8") as handle:
            lines = [line for line in handle.read().split("\n") if line]
        start = 0
        for index in range(len(lines) - 1, -1, -1):
            record = json.loads(lines[index])
            ops: List[Op] = []
            for write in record.get("writes") or []:
                if write.get("type") == "doc":
                    ops.extend(tuple(op) for op in write.get("ops") or [])
            if is_base(ops):
                start = index
                break
        if start == 0:
            return
        tmp = f"{file}.tmp"
        with open(tmp, "wb") as handle:
            handle.write("".join(f"{line}\n" for line in lines[start:]).encode("utf-8"))
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        old = self._sidecars.pop(ref.conversation_id, None)
        if old is not None:
            old.close()
        os.replace(tmp, file)
        self._sidecars[ref.conversation_id] = open(file, "ab")

    async def close(self, ctx: Optional[Context] = None) -> None:
        if self._closed_once:
            return
        self._closed_once = True
        await MemoryStorage.close(self)
        self._main.close()
        for handle in self._sidecars.values():
            handle.close()
        for handle in self._task_sidecars.values():
            handle.close()
        self._sidecars.clear()
        self._task_sidecars.clear()


_ = (JsonObject, Storage)
