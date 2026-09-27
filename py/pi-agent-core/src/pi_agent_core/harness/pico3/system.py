"""System sections, the managed entry and preparation ported from ``harness/pico3/system.ts``.

Port note: the TypeScript draft's generic ``SystemSection<T>`` is erased in
Python (``JsonValue``); the renderer receives the raw JSON value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from ..._chord.context import Context
from pi_ai.types import JsonObject, JsonValue
from .types import (
    AnyToolDeclaration,
    ContextEdit,
    Entry,
    HookInfo,
    HookResult,
    Id,
    RewindableState,
    Runtime,
    Stored,
    SystemMessage,
    to_stored,
)

__all__ = [
    "SystemSection",
    "define_system_section",
    "system_sections",
    "EnvironmentInfo",
    "SkillInfo",
    "SectionSeed",
    "section_seed",
    "remove_section",
    "SectionRecord",
    "SystemEntryData",
    "SectionState",
    "Canonical",
    "FoldResult",
    "fold_canonical",
    "SystemSectionDraft",
    "Draft",
    "SectionRegistry",
    "ToolRegistry",
    "PreparationSnapshot",
    "same_snapshot",
    "TakeSnapshotResult",
    "take_snapshot",
    "SystemInstructionsHooks",
    "PrepareDraftResult",
    "prepare_draft",
    "ManagedEntryPlan",
    "plan_managed_entry",
    "effective_tools",
    "DEFAULT_RETRY_POLICY",
]

#: Declared retry default for the sticky document (owned by the generation kind).
DEFAULT_RETRY_POLICY: Dict[str, JsonValue] = {
    "enabled": True,
    "maxRetries": 2,
    "baseDelayMs": 1000,
}


# ---------------------------------------------------------------------------
# Sections (pico §12.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemSection:
    """A stable key plus a pure renderer. Only keys, payloads and rendered text are stored."""

    key: str
    render: Callable[[Any], str]


def define_system_section(key: str, render: Callable[[Any], str]) -> SystemSection:
    return SystemSection(key=key, render=render)


@dataclass
class EnvironmentInfo:
    cwd: str = ""


@dataclass
class SkillInfo:
    name: str = ""
    description: str = ""


class _SystemSections:
    """The built-in section registry (``systemSections`` in TypeScript)."""

    def __init__(self) -> None:
        self.identity = define_system_section("identity", lambda v: v)
        self.environment = define_system_section(
            "environment", lambda v: f"Working directory: {_field(v, 'cwd')}"
        )
        self.skills = define_system_section("skills", _render_skills)

    def __iter__(self):
        return iter((self.identity, self.environment, self.skills))


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _render_skills(value: Any) -> str:
    items = value or []
    parts = []
    for entry in items:
        parts.append(f"- {_field(entry, 'name')}: {_field(entry, 'description')}")
    return "\n".join(parts)


system_sections = _SystemSections()


@dataclass
class SectionSeed:
    """Section seed for a new conversation (§8.1): set with a checked pair, or remove an inherited key."""

    key: str
    value: JsonValue = None
    remove: bool = False

    def to_json(self) -> JsonObject:
        if self.remove:
            return {"key": self.key, "remove": True}
        return {"key": self.key, "value": self.value}


def section_seed(section: SystemSection, value: JsonValue) -> SectionSeed:
    return SectionSeed(key=section.key, value=value)


def remove_section(key: str) -> SectionSeed:
    return SectionSeed(key=key, remove=True)


@dataclass
class SectionRecord:
    key: str
    action: str = "set"  # "set" | "remove"
    value: JsonValue = None
    rendered: str = ""

    def to_json(self) -> JsonObject:
        if self.action == "remove":
            return {"key": self.key, "action": "remove"}
        return {"key": self.key, "action": "set", "value": self.value, "rendered": self.rendered}

    @staticmethod
    def from_json(data: JsonObject) -> "SectionRecord":
        if data.get("action") == "remove":
            return SectionRecord(key=str(data.get("key")), action="remove")
        return SectionRecord(
            key=str(data.get("key")),
            action="set",
            value=data.get("value"),
            rendered=str(data.get("rendered", "")),
        )


@dataclass
class SystemEntryData:
    sections: List[SectionRecord] = field(default_factory=list)
    baseline: bool = False

    def to_json(self) -> JsonObject:
        data: JsonObject = {"sections": [record.to_json() for record in self.sections]}
        if self.baseline:
            data["baseline"] = True
        return data

    @staticmethod
    def from_json(data: JsonObject) -> "SystemEntryData":
        raw = data.get("sections") or []
        return SystemEntryData(
            sections=[SectionRecord.from_json(item) for item in raw],
            baseline=bool(data.get("baseline")),
        )


@dataclass
class SectionState:
    value: JsonValue
    rendered: str


#: Insertion order is canonical order.
Canonical = Dict[str, SectionState]


@dataclass
class FoldResult:
    canonical: Canonical
    newest_managed: Optional[Id] = None
    newest_baseline: Optional[Id] = None


async def fold_canonical(reads: Any, conversation_id: Id) -> FoldResult:
    """Fold the fork-visible managed entries from the newest baseline forward."""
    from .types import EntryScan

    managed: List[Entry] = []
    before: Optional[Id] = None
    baseline: Optional[Id] = None
    while True:
        scan = EntryScan(conversation_id=conversation_id, kind="pi.system", limit=64, before=before)
        page = await reads.scan_entries(scan)
        for entry in page:
            managed.append(entry)
            data = SystemEntryData.from_json(entry.data or {})
            if data.baseline:
                baseline = entry.id
                break
        else:
            if len(page) < 64:
                break
            before = page[-1].id
            continue
        break
    managed.reverse()
    canonical: Canonical = {}
    for entry in managed:
        data = SystemEntryData.from_json(entry.data or {})
        for record in data.sections:
            if record.action == "remove":
                canonical.pop(record.key, None)
            else:
                canonical[record.key] = SectionState(value=record.value, rendered=record.rendered)
    newest_managed = managed[-1].id if managed else None
    return FoldResult(canonical=canonical, newest_managed=newest_managed, newest_baseline=baseline)


# ---------------------------------------------------------------------------
# Draft (§12.2)
# ---------------------------------------------------------------------------


class SystemSectionDraft(Protocol):
    """What ``systemInstructions`` handlers edit."""

    def get(self, section: SystemSection) -> Optional[JsonValue]: ...

    def set(self, section: SystemSection, value: JsonValue) -> None: ...

    def delete(self, section: SystemSection) -> None: ...

    def wrap(self, section: SystemSection, transform: Callable[[str], str]) -> None: ...


@dataclass
class DraftSnapshot:
    values: Dict[str, JsonValue] = field(default_factory=dict)
    wrappers: Dict[str, List[Callable[[str], str]]] = field(default_factory=dict)
    touched: set = field(default_factory=set)


class Draft:
    """The mutable section draft handed to ``systemInstructions`` handlers."""

    def __init__(self, seed: Canonical) -> None:
        self.values: Dict[str, JsonValue] = {key: state.value for key, state in seed.items()}
        self.wrappers: Dict[str, List[Callable[[str], str]]] = {}
        self.touched: set = set()

    def get(self, section: SystemSection) -> Optional[JsonValue]:
        if section.key not in self.values:
            return None
        return json.loads(json.dumps(self.values[section.key]))

    def set(self, section: SystemSection, value: JsonValue) -> None:
        self.values[section.key] = value
        self.touched.add(section.key)

    def delete(self, section: SystemSection) -> None:
        self.values.pop(section.key, None)
        self.wrappers.pop(section.key, None)
        self.touched.add(section.key)

    def wrap(self, section: SystemSection, transform: Callable[[str], str]) -> None:
        self.wrappers.setdefault(section.key, []).append(transform)
        self.touched.add(section.key)

    def snapshot(self) -> DraftSnapshot:
        return DraftSnapshot(
            values=dict(self.values),
            wrappers={key: list(value) for key, value in self.wrappers.items()},
            touched=set(self.touched),
        )

    def restore(self, snapshot: DraftSnapshot) -> None:
        self.values = snapshot.values
        self.wrappers = snapshot.wrappers
        self.touched = snapshot.touched


def freeze(
    draft: Draft,
    canonical: Canonical,
    registry: Dict[str, SystemSection],
) -> Canonical:
    """Render touched sections; untouched keep their stored text."""
    desired: Canonical = {}
    for key, value in draft.values.items():
        if key not in draft.touched:
            previous = canonical.get(key)
            if previous is not None:
                desired[key] = previous
            continue
        definition = registry.get(key)
        if definition is None:
            continue  # unregistered: cannot re-render; treated as removed
        rendered = definition.render(value)
        for wrapper in draft.wrappers.get(key, []):
            rendered = wrapper(rendered)
        desired[key] = SectionState(value=value, rendered=rendered)
    return desired


# ---------------------------------------------------------------------------
# Preparation (§12.5)
# ---------------------------------------------------------------------------


@dataclass
class SectionRegistry:
    map: Dict[str, SystemSection] = field(default_factory=dict)
    revision: int = 0


@dataclass
class ToolRegistry:
    map: Dict[str, AnyToolDeclaration] = field(default_factory=dict)
    revision: int = 0


@dataclass
class PreparationSnapshot:
    """Snapshot S. Compared field by field before the ``prepared`` commit."""

    newest_managed: Optional[Id] = None
    newest_baseline: Optional[Id] = None
    newest_head: Optional[Id] = None
    settings: Dict[str, JsonValue] = field(default_factory=dict)
    sections_rev: int = 0
    tools_rev: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "newestManaged": self.newest_managed,
            "newestBaseline": self.newest_baseline,
            "newestHead": self.newest_head,
            "settings": self.settings,
            "sectionsRev": self.sections_rev,
            "toolsRev": self.tools_rev,
        }


def same_snapshot(a: PreparationSnapshot, b: PreparationSnapshot) -> bool:
    return json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


@dataclass
class TakeSnapshotResult:
    snapshot: PreparationSnapshot
    canonical: Canonical
    seed: Optional[List[SectionSeed]] = None


async def take_snapshot(
    tx: Any,
    conversation_id: Id,
    sections: SectionRegistry,
    tools: ToolRegistry,
) -> TakeSnapshotResult:
    from .types import DocRef

    folded = await fold_canonical(tx, conversation_id)
    head = await tx.newest_entry(conversation_id, {"withHead": True})
    r: RewindableState = tx.snapshot(DocRef(doc="rewindable", conversation_id=conversation_id))
    conv = await tx.conversation(conversation_id)
    settings: Dict[str, JsonValue] = {
        "thinkingLevel": r.get("thinkingLevel"),
        "selectedTools": list(r.get("selectedTools") or []),
        "profile": r.get("profile"),
    }
    if r.get("model") is not None:
        settings["model"] = r["model"]
    snapshot = PreparationSnapshot(
        newest_managed=folded.newest_managed,
        newest_baseline=folded.newest_baseline,
        newest_head=head.id if head is not None else None,
        settings=settings,
        sections_rev=sections.revision,
        tools_rev=tools.revision,
    )
    seed = _conversation_seeds(conv) if folded.newest_managed is None else None
    return TakeSnapshotResult(snapshot=snapshot, canonical=folded.canonical, seed=seed)


def _conversation_seeds(conversation: Any) -> Optional[List[SectionSeed]]:
    raw = getattr(conversation, "sections", None)
    if not raw:
        return None
    seeds: List[SectionSeed] = []
    for item in raw:
        if isinstance(item, SectionSeed):
            seeds.append(item)
        elif isinstance(item, dict) and item.get("remove"):
            seeds.append(remove_section(str(item.get("key"))))
        else:
            seeds.append(SectionSeed(key=str(item.get("key")), value=item.get("value")))
    return seeds


class SystemInstructionsHooks(Protocol):
    """Handlers that edit the draft and may override the tool loadout."""

    async def system_instructions(
        self, input: Dict[str, Any], info: HookInfo, ctx: Context
    ) -> HookResult: ...


@dataclass
class PrepareDraftResult:
    desired: Canonical
    tools: List[AnyToolDeclaration]


async def prepare_draft(
    rt: Any,
    sections: SectionRegistry,
    canonical: Canonical,
    seed: Optional[Sequence[SectionSeed]],
    settings: Dict[str, JsonValue],
    on_report: Callable[[str], None],
    ctx: Context,
) -> PrepareDraftResult:
    """Off the line: seed the draft, run handlers, freeze."""
    draft = Draft(canonical)
    for item in seed or []:
        if isinstance(item, dict):
            item = (
                remove_section(str(item.get("key")))
                if item.get("remove")
                else SectionSeed(key=str(item.get("key")), value=item.get("value"))
            )
        if item.remove:
            draft.delete(SystemSection(key=item.key, render=lambda _value: ""))
        else:
            draft.set(SystemSection(key=item.key, render=lambda _value: ""), item.value)

    default_tools: List[AnyToolDeclaration] = []
    for name in settings.get("selectedTools") or []:
        tool = rt.tools.get(name)
        if tool is not None:
            default_tools.append(tool)
        else:
            on_report(f"selected tool {name} is not registered")
    tools: List[AnyToolDeclaration] = list(default_tools)
    for binding in rt.hooks.handlers():
        before = draft.snapshot()
        try:
            handler = getattr(binding.handlers, "system_instructions", None)
            if handler is None:
                continue
            out = handler(
                {"sections": draft, "config": settings, "tools": default_tools}, binding.api, ctx
            )
            if hasattr(out, "__await__"):
                out = await out
            if isinstance(out, dict) and out.get("tools"):
                tools = list(out["tools"])
        except Exception as error:  # noqa: BLE001 - reported, not re-raised
            if ctx.abort_signal is not None and ctx.abort_signal.aborted:
                raise
            draft.restore(before)
            on_report(str(error))
    return PrepareDraftResult(desired=freeze(draft, canonical, sections.map), tools=tools)


@dataclass
class ManagedEntryPlan:
    data: SystemEntryData
    model: List[Stored[Any]]
    edits: Optional[List[ContextEdit]] = None


def _tool_of(tool: AnyToolDeclaration) -> JsonObject:
    """The stored form of one tool declaration: a declaration object or an already-stored tool."""
    if isinstance(tool, dict):
        return to_stored(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            }
        )
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "parameters": to_stored(getattr(tool, "parameters", None)),
    }


async def plan_managed_entry(
    tx: Any,
    conversation_id: Id,
    snapshot: PreparationSnapshot,
    canonical: Canonical,
    desired: Canonical,
    tools: Sequence[AnyToolDeclaration],
    now: int,
) -> Optional[ManagedEntryPlan]:
    """On the line, in the ``prepared`` commit: diff desired against canonical."""
    need_baseline = snapshot.newest_managed is None or (
        snapshot.newest_head is not None
        and (snapshot.newest_baseline is None or snapshot.newest_head > snapshot.newest_baseline)
    )

    view = await tx.context(conversation_id)
    previous: Dict[str, Any] = {}
    for message in view.messages:
        if _message_role(message) != "system":
            continue
        for tool in _message_field(message, "toolsRemoved") or []:
            previous.pop(_field(tool, "name"), None)
        for tool in _message_field(message, "toolsAdded") or []:
            previous[_field(tool, "name")] = tool

    if need_baseline:
        records = [
            SectionRecord(key=key, action="set", value=state.value, rendered=state.rendered)
            for key, state in desired.items()
        ]
        content = "\n\n".join(
            f"## {record.key}\n{record.rendered}" for record in records if record.action == "set"
        )
        entries = view.entries
        edits = [
            ContextEdit(target=entry.id, action="omit")
            for entry in entries
            if entry.kind == "pi.system" and entry.id > (snapshot.newest_head or 0)
        ]
        message = SystemMessage(
            content=content,
            tools_added=[_tool_of(tool) for tool in tools],
            timestamp=now,
        )
        return ManagedEntryPlan(
            data=SystemEntryData(sections=records, baseline=True),
            model=[message.to_json()],
            edits=edits or None,
        )

    changed: List[SectionRecord] = []
    for key, state in desired.items():
        previous_state = canonical.get(key)
        if (
            previous_state is None
            or previous_state.rendered != state.rendered
            or json.dumps(previous_state.value, sort_keys=True) != json.dumps(state.value, sort_keys=True)
        ):
            changed.append(
                SectionRecord(key=key, action="set", value=state.value, rendered=state.rendered)
            )
    for key in canonical:
        if key not in desired:
            changed.append(SectionRecord(key=key, action="remove"))
    added = [tool for tool in tools if _field(tool, "name") not in previous]
    removed = [
        value
        for value in previous.values()
        if not any(_field(tool, "name") == _field(value, "name") for tool in tools)
    ]
    if len(changed) == 0 and len(added) == 0 and len(removed) == 0:
        return None

    render_changed = any(
        record.action == "remove" or canonical.get(record.key) is None or canonical[record.key].rendered != record.rendered
        for record in changed
    )
    if not render_changed and len(added) == 0 and len(removed) == 0:
        return ManagedEntryPlan(data=SystemEntryData(sections=changed), model=[])

    content = "\n\n".join(
        f"The {record.key} section now reads:\n{record.rendered}"
        if record.action == "set"
        else f"The {record.key} section no longer applies."
        for record in changed
    )
    message = SystemMessage(content=content, timestamp=now)
    if added:
        message.tools_added = [_tool_of(tool) for tool in added]
    if removed:
        message.tools_removed = list(removed)
    return ManagedEntryPlan(data=SystemEntryData(sections=changed), model=[message.to_json()])


def _message_role(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def effective_tools(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Fold ``toolsAdded``/``toolsRemoved`` across a projected request's SystemMessages."""
    tools: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        if _message_role(message) != "system":
            continue
        for tool in _message_field(message, "toolsRemoved") or []:
            tools.pop(_field(tool, "name"), None)
        for tool in _message_field(message, "toolsAdded") or []:
            tools[_field(tool, "name")] = tool
    return list(tools.values())


_ = Runtime
