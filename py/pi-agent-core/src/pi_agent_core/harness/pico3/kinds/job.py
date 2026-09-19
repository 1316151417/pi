"""Job task kind ported from ``harness/pico3/kinds/job.ts``.

Runs a process through the process host and reports its exit code, with an
optional repeat interval and an abort path that escalates SIGTERM to SIGKILL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...._chord.context import Context
from pi_ai.types import JsonValue
from ..types import Completion, Id, Kind, ProcessStatus, Step, Task, TaskRef

__all__ = [
    "JobInput",
    "JobCheckpoint",
    "JobOutput",
    "JobResult",
    "JobFailure",
    "job",
    "job_kind",
]


@dataclass
class JobInput:
    command: str = ""
    args: Optional[list] = None
    cwd: str = ""
    env: Optional[Dict[str, str]] = None
    notify: bool = False
    rerun: bool = False
    every: Optional[int] = None
    not_before: Optional[int] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "command": self.command,
            "cwd": self.cwd,
            "notify": self.notify,
            "rerun": self.rerun,
        }
        if self.args is not None:
            data["args"] = list(self.args)
        if self.env is not None:
            data["env"] = dict(self.env)
        if self.every is not None:
            data["every"] = self.every
        if self.not_before is not None:
            data["notBefore"] = self.not_before
        return data


@dataclass
class JobCheckpoint:
    phase: str = "waiting"  # "waiting" | "spawning" | "running"
    until_ms: Optional[int] = None
    key: Optional[str] = None
    occurrence: int = 1


@dataclass
class JobOutput:
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    dropped_stdout: Optional[int] = None
    dropped_stderr: Optional[int] = None
    exit_code: Optional[int] = None
    occurrence: Optional[int] = None


@dataclass
class JobResult:
    exitCode: int = 0
    occurrences: int = 0
    stdout: str = ""
    stderr: str = ""

    def to_json(self) -> JsonObject:
        return {
            "exitCode": self.exitCode,
            "occurrences": self.occurrences,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class JobFailure:
    reason: str = "spawn"  # "spawn" | "interrupted"
    detail: str = ""

    def to_json(self) -> JsonObject:
        return {"reason": self.reason, "detail": self.detail}


def _failed(reason: str, detail: str) -> Completion:
    return Completion(status="failed", failure=JobFailure(reason=reason, detail=detail))


def _notice(text: str, now: int) -> Dict[str, Any]:
    return {"kind": "pi.notice", "model": [{"role": "user", "content": text, "timestamp": now}]}


def _input(task: Task) -> Dict[str, Any]:
    if isinstance(task.input, dict):
        return task.input
    return dict(task.input.__dict__) if task.input is not None else {}


def _checkpoint(task: Task) -> Dict[str, Any]:
    return dict(task.checkpoint or {})


async def _spawn(task: Task, occurrence: int, runtime: Any, ctx: Context) -> Step:
    input_record = _input(task)
    key = f"{task.id}:{occurrence}"
    await runtime.commit(
        lambda tx, current=None, _ctx=None: tx.checkpoint(
            {"phase": "spawning", "key": key, "occurrence": occurrence}
        ),
        ctx,
    )
    try:
        await runtime.process_host.start(key, input_record, ctx)
    except Exception as error:  # noqa: BLE001 - reported as a task failure
        if ctx.signal is not None and ctx.signal.aborted:
            raise

        async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
            if input_record.get("notify"):
                await tx.write(
                    current.conversation_id,
                    _notice(f"job {task.id} failed to start: {error}", runtime.now()),
                )
            return _failed("spawn", str(error))

        return Step(done=done)
    await runtime.commit(
        lambda tx, current=None, _ctx=None: tx.checkpoint(
            {"phase": "running", "key": key, "occurrence": occurrence}
        ),
        ctx,
    )
    return await _poll(task, key, occurrence, runtime, ctx)


async def _reconcile(task: Task, checkpoint: Dict[str, Any], runtime: Any, ctx: Context) -> Step:
    if runtime.process_host is None:
        return Step(done=lambda _tx=None, _task=None: _failed("interrupted", "no process host after restart"))
    try:
        status: ProcessStatus = await runtime.process_host.status(checkpoint["key"], ctx)
    except Exception as error:  # noqa: BLE001 - reported as a task failure
        return Step(
            done=lambda _tx=None, _task=None, error=error: _failed(
                "interrupted", f"host status failed: {error}"
            )
        )
    if status.status == "unknown":
        if not _input(task).get("rerun"):
            return Step(done=lambda _tx=None, _task=None: _failed("interrupted", "process outcome unknown"))
        return await _spawn(task, checkpoint["occurrence"], runtime, ctx)
    if checkpoint["phase"] == "spawning":
        await runtime.commit(
            lambda tx, current=None, _ctx=None, checkpoint=checkpoint: tx.checkpoint(
                {
                    "phase": "running",
                    "key": checkpoint["key"],
                    "occurrence": checkpoint["occurrence"],
                }
            ),
            ctx,
        )
    return await _poll(task, checkpoint["key"], checkpoint["occurrence"], runtime, ctx)


async def _poll(task: Task, key: str, occurrence: int, runtime: Any, ctx: Context) -> Step:
    input_record = _input(task)
    attempt = 0
    while True:
        try:
            status: ProcessStatus = await runtime.process_host.status(key, ctx)
        except Exception as error:  # noqa: BLE001 - reported as a task failure
            return Step(
                done=lambda _tx=None, _task=None, error=error: _failed(
                    "interrupted", f"host status failed: {error}"
                )
            )
        if status.status == "unknown":
            if not input_record.get("rerun"):
                return Step(done=lambda _tx=None, _task=None: _failed("interrupted", "process outcome unknown"))
            return await _spawn(task, occurrence, runtime, ctx)

        snapshot = status

        def write_slot(
            tx: Any,
            current: Optional[Task] = None,
            _ctx: Any = None,
            snapshot: ProcessStatus = snapshot,
        ) -> None:
            output = tx.slot(TaskRef(id=current.id, kind=job_kind))
            output["stdout"] = snapshot.stdout
            output["stderr"] = snapshot.stderr
            output["droppedStdout"] = snapshot.dropped_stdout
            output["droppedStderr"] = snapshot.dropped_stderr
            if snapshot.status == "exited":
                output["exitCode"] = snapshot.exit_code

        await runtime.commit(write_slot, ctx)
        if snapshot.status == "exited":
            exit_code = snapshot.exit_code
            stdout = snapshot.stdout
            stderr = snapshot.stderr
            if input_record.get("every") is None:

                async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
                    if input_record.get("notify"):
                        await tx.write(
                            current.conversation_id,
                            _notice(f"job {task.id} exited with code {exit_code}", runtime.now()),
                        )
                    return Completion(
                        status="completed",
                        result=JobResult(
                            exitCode=exit_code,
                            occurrences=occurrence,
                            stdout=stdout,
                            stderr=stderr,
                        ),
                    )

                return Step(done=done)

            until_ms = runtime.now() + input_record["every"]

            async def next_checkpoint(
                tx: Any, current: Optional[Task] = None, _ctx: Any = None
            ) -> Dict[str, Any]:
                if input_record.get("notify"):
                    await tx.write(
                        current.conversation_id,
                        _notice(
                            f"job {task.id} occurrence {occurrence} exited with code {exit_code}",
                            runtime.now(),
                        ),
                    )
                output = tx.slot(TaskRef(id=current.id, kind=job_kind))
                output["stdout"] = ""
                output["stderr"] = ""
                output["droppedStdout"] = 0
                output["droppedStderr"] = 0
                output.pop("exitCode", None)
                output["occurrence"] = occurrence + 1
                return {"phase": "waiting", "untilMs": until_ms, "occurrence": occurrence + 1}

            return Step(next=next_checkpoint)

        attempt += 1
        await runtime.sleep(runtime.now() + min(1000, 100 * 2 ** min(attempt - 1, 4)), ctx)


async def _waiting(task: Task, runtime: Any, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    await runtime.sleep(checkpoint["untilMs"], ctx)
    return await _spawn(task, checkpoint["occurrence"], runtime, ctx)


async def _spawning(task: Task, runtime: Any, ctx: Context) -> Step:
    return await _reconcile(task, _checkpoint(task), runtime, ctx)


async def _running(task: Task, runtime: Any, ctx: Context) -> Step:
    return await _reconcile(task, _checkpoint(task), runtime, ctx)


async def _initial(task: Task, runtime: Any, ctx: Context) -> Step:
    if runtime.process_host is None:
        return Step(done=lambda _tx=None, _task=None: _failed("spawn", "no process host"))
    now = runtime.now()
    input_record = _input(task)
    until_ms = input_record.get("notBefore")
    if until_ms is None:
        until_ms = now
    if until_ms > now:
        return Step(next={"phase": "waiting", "untilMs": until_ms, "occurrence": 1})
    return await _spawn(task, 1, runtime, ctx)


async def _abort(task: Task, runtime: Any, ctx: Context) -> Any:
    checkpoint = _checkpoint(task)
    if not checkpoint or checkpoint.get("phase") == "waiting" or runtime.process_host is None:
        return lambda *_args: {"killed": False}
    await runtime.process_host.kill(checkpoint["key"], "SIGTERM", ctx)
    await runtime.sleep(runtime.now() + 5000, ctx)
    await runtime.process_host.kill(checkpoint["key"], "SIGKILL", ctx)
    return lambda *_args: {"killed": True}


def _describe(task: Any) -> Dict[str, Any]:
    """Public slot for the job: its stage plus whatever the host wrote."""
    if isinstance(task, dict):
        stage = (task.get("checkpoint") or {}).get("phase") or "starting"
        slot = task.get("slot") or {}
    else:
        stage = _checkpoint(task).get("phase") or "starting"
        slot = task.slot or {}
    return {"stage": stage, **slot}


job_kind = Kind(
    name="pi.job",
    slot=lambda _task: {},
    describe=_describe,
    inflight=["spawning", "running"],
    phases={"waiting": _waiting, "spawning": _spawning, "running": _running},
    initial=_initial,
    abort=_abort,
)

#: Frozen declaration, mirroring ``Object.freeze(jobKind)``.
job = job_kind


_ = (Id, JsonValue)
