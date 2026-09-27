"""TypeBox 1.3.27 schema resource and reference traversal (MIT; see licenses)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote

from ada_url import URL

from ._typebox_primitives import entries, get_property, has_property, schema_keyword
from ._values import UNDEFINED

DEFAULT_BASE = "https://json-schema.org"


def _url(value: str, base: str = DEFAULT_BASE) -> URL:
    value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    return URL(value, base)


def _decode(value: str) -> str:
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError("URI malformed")
    return unquote(value, errors="strict")


def _pointer(value: object, pointer: str) -> object:
    components = pointer.split("/")
    if components and components[0] == "":
        components.pop(0)
    for component in components:
        key = component.replace("~1", "/").replace("~0", "~")
        if key in ("__proto__", "constructor", "prototype"):
            return UNDEFINED
        if isinstance(value, dict):
            value = get_property(value, key)
        elif isinstance(value, list):
            if key == "length":
                value = len(value)
            elif key == "0" or (key.isascii() and key.isdecimal() and not key.startswith("0") and len(key) < 11):
                index = int(key)
                value = value[index] if index < len(value) else UNDEFINED
            else:
                return UNDEFINED
        else:
            return UNDEFINED
    return value


def _hash(schema: object, ref: URL) -> object:
    if ref.href.endswith("#"):
        return schema
    if not ref.hash.startswith("#"):
        return UNDEFINED
    fragment = _decode(ref.hash[1:])
    return _pointer(schema, fragment) if fragment.startswith("/") else UNDEFINED


def _from_value(schema: object, base: URL, ref: URL) -> object:
    next_base = _url(str(schema["$id"]), base.href) if schema_keyword(schema, "$id") else base
    if isinstance(schema, dict):
        if schema_keyword(schema, "$id"):
            if schema["$id"] == ref.hash:
                return schema
            if next_base.pathname == _url(ref.href, next_base.href).pathname:
                result = _hash(schema, ref) if ref.hash.startswith("#") else schema
                if result is not UNDEFINED:
                    return result
        for key in ("$anchor", "$dynamicAnchor"):
            if schema_keyword(schema, key) and _url("#" + str(schema[key]), next_base.href).href == _url(ref.href, next_base.href).href:
                return schema
        result = _hash(schema, ref)
        if result is not UNDEFINED:
            return result
    result: object = UNDEFINED
    for key, child in entries(schema):
        if isinstance(schema, dict) and key in ("const", "enum"):
            continue
        match = _from_value(child, next_base, ref)
        if match is not UNDEFINED:
            result = match
    return result


def _find_base(schema: object, base: URL, target: object) -> object:
    if schema is target:
        return base.href
    next_base = _url(str(schema["$id"]), base.href) if schema_keyword(schema, "$id") else base
    for _, child in entries(schema):
        result = _find_base(child, next_base, target)
        if result is not UNDEFINED:
            return result
    return UNDEFINED


def _find_anchor(schema: object, name: str) -> object:
    if schema_keyword(schema, "$dynamicAnchor") and schema["$dynamicAnchor"] == name:
        return schema
    for _, child in entries(schema):
        result = _find_anchor(child, name)
        if result is not UNDEFINED:
            return result
    return UNDEFINED


def resolve_ref(context: dict[str, object], schema: object, base: str, ref: str, apply_schema_id: bool = True) -> object:
    initial_base = _url(base or ".")
    resolved_base = _url(str(schema["$id"]), initial_base.href) if apply_schema_id and schema_keyword(schema, "$id") else initial_base
    initial_ref = _url(ref, resolved_base.href)
    direct = get_property(context, ref) if has_property(context, ref) else UNDEFINED
    if direct is not UNDEFINED and direct is not None:
        return direct
    local = _from_value(schema, resolved_base, initial_ref)
    if local is not UNDEFINED and local is not None:
        return local
    canonical = initial_ref.href.split("#")[0]
    if canonical == resolved_base.href.split("#")[0] or canonical not in context:
        return UNDEFINED
    remote = context[canonical]
    remote_base = _url(str(remote["$id"]), canonical) if schema_keyword(remote, "$id") else _url(canonical)
    return remote if initial_ref.hash == "" else _from_value(remote, remote_base, initial_ref)


@dataclass
class _Frame:
    schema: object
    base: str
    id_depth: int
    resource_depth: int


class ReferenceStack:
    def __init__(self, schema: object, context: dict[str, object] | None = None) -> None:
        self.context = {} if context is None else context
        self.schema = schema
        self.ids: list[dict[str, object]] = []
        self.resource_ids: list[dict[str, object]] = []
        self.anchors: list[dict[str, object]] = []
        self.recursive_anchors: list[dict[str, object]] = []
        self.dynamic_anchors: list[dict[str, object]] = []
        self.retrieved_resources: dict[int, tuple[object, str]] = {}
        self.retrieved_frames: list[_Frame] = []
        self.resolved_resources: dict[int, dict[str, object]] = {}
        self.pending_resource = True

    def _resource_anchors(self, schema: object, add: bool, root: bool = True) -> None:
        if not isinstance(schema, (dict, list)):
            return
        if not root and schema_keyword(schema, "$id"):
            return
        if not root and schema_keyword(schema, "$dynamicAnchor"):
            if add:
                self.dynamic_anchors.append(schema)
            else:
                self.dynamic_anchors.pop()
        for _, child in entries(schema):
            self._resource_anchors(child, add, False)

    def _register(self, schema: dict[str, object]) -> None:
        self.ids.append(schema)
        is_resource = self.pending_resource
        self.pending_resource = False
        if is_resource:
            self.resource_ids.append(schema)
        self._resource_anchors(schema, True)

    def _unregister(self, schema: dict[str, object]) -> None:
        self.ids.pop()
        if self.resource_ids and self.resource_ids[-1] is schema:
            self.resource_ids.pop()
        self._resource_anchors(schema, False)

    def push(self, schema: object) -> None:
        if not isinstance(schema, dict):
            return
        if schema_keyword(schema, "$id"):
            self._register(schema)
        if schema_keyword(schema, "$anchor"):
            self.anchors.append(schema)
        if schema.get("$recursiveAnchor") is True:
            self.recursive_anchors.append(schema)
        if schema_keyword(schema, "$dynamicAnchor"):
            self.dynamic_anchors.append(schema)
        retrieved = self.retrieved_resources.get(id(schema))
        if retrieved is not None:
            root, base = retrieved
            self.retrieved_frames.append(_Frame(root, base, len(self.ids), len(self.resource_ids)))

    def pop(self, schema: object) -> None:
        if not isinstance(schema, dict):
            return
        if schema_keyword(schema, "$id"):
            self._unregister(schema)
        if schema_keyword(schema, "$anchor"):
            self.anchors.pop()
        if schema.get("$recursiveAnchor") is True:
            self.recursive_anchors.pop()
        if schema_keyword(schema, "$dynamicAnchor"):
            self.dynamic_anchors.pop()
        if id(schema) in self.retrieved_resources:
            self.retrieved_frames.pop()
        resource = self.resolved_resources.pop(id(schema), None)
        if resource is not None:
            self._unregister(resource)

    def _base(self, stack: list[dict[str, object]]) -> str:
        frame = self.retrieved_frames[-1] if self.retrieved_frames else None
        base = _url(frame.base if frame else DEFAULT_BASE)
        for schema in stack[frame.id_depth:] if frame else stack:
            base = _url(str(schema["$id"]), base.href)
        return base.href

    def _resource_base(self) -> str:
        if not self.retrieved_frames:
            return self._base(self.resource_ids)
        frame = self.retrieved_frames[-1]
        base = _url(frame.base)
        for schema in self.resource_ids[frame.resource_depth:]:
            base = _url(str(schema["$id"]), base.href)
        return base.href

    def _reference_base(self) -> str:
        if self.retrieved_frames:
            return self._resource_base()
        if self.ids and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", str(self.ids[-1]["$id"])):
            return self._base(self.ids)
        return self._resource_base()

    def _lexical(self) -> object:
        if self.retrieved_frames:
            frame = self.retrieved_frames[-1]
            return self.ids[-1] if len(self.ids) > frame.id_depth else frame.schema
        return self.ids[-1] if self.ids else self.schema

    def ref(self, schema: dict[str, object], keyword: str) -> object:
        ref = str(schema[keyword])
        lexical = self._lexical()
        if keyword == "$recursiveRef":
            root = self.recursive_anchors[0] if isinstance(lexical, dict) and lexical.get("$recursiveAnchor") is True else lexical
            target = resolve_ref(self.context, root, self._base(self.ids), ref, False)
        elif keyword == "$dynamicRef":
            base = self._base(self.ids)
            root = lexical if ref.startswith("#") else self.schema
            target = resolve_ref(self.context, root, base, ref, False)
            fragment = _url(ref, base).hash
            if target is UNDEFINED:
                if fragment.startswith("#") and not fragment.startswith("#/"):
                    name = _decode(fragment[1:])
                    target = next((anchor for anchor in self.dynamic_anchors if anchor.get("$dynamicAnchor") == name), UNDEFINED)
                    if target is UNDEFINED:
                        target = _find_anchor(self.schema, name)
            elif schema_keyword(target, "$dynamicAnchor") and not fragment.startswith("#/"):
                name = target["$dynamicAnchor"]
                target = next((anchor for anchor in self.dynamic_anchors if anchor.get("$dynamicAnchor") == name), target)
        else:
            source = lexical if self.retrieved_frames else self.schema
            root = lexical if ref.startswith("#") else source
            base = self._reference_base()
            target = resolve_ref(self.context, root, base, ref, False)
            if target is not UNDEFINED:
                self.pending_resource = True
            if not isinstance(target, dict):
                return target
            canonical = _url(ref, base).href.split("#")[0]
            remote = canonical != self._resource_base()
            retrieved: tuple[object, str] | None = None
            if remote and isinstance(self.context.get(canonical), dict):
                retrieved = self.context[canonical], canonical
            elif ref.startswith("#") and (not isinstance(self.schema, dict) or "$schema" not in self.schema):
                target_base = _find_base(lexical, _url(base or "."), target)
                if isinstance(target_base, str) and target_base != base:
                    retrieved = lexical, target_base
            if remote and not schema_keyword(target, "$id"):
                resource = resolve_ref(self.context, self.schema, base, canonical, False)
                if schema_keyword(resource, "$id") and not any(resource is item for item in self.ids):
                    self._register(resource)
                    self.resolved_resources[id(target)] = resource
            if retrieved is not None:
                self.retrieved_resources[id(target)] = retrieved
            return target
        if target is not UNDEFINED and target is not False and target is not None:
            self.pending_resource = True
        return target
