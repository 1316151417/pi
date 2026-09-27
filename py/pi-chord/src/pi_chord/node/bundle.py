"""Content-addressed CommonJS bundling from ``chord/src/node/bundle.ts``."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from icu import Collator

from .._undefined import UNDEFINED, Undefined
from ._esbuild import EsbuildFailure, EsbuildService
from .bundle_loader import _manifest_json, _object_keys, _utf8
from .manifest import FACET_BUNDLE_MANIFEST_FILE, FacetBundleEntry, FacetBundleManifest, FacetBundlePlugin

type FacetBundlePlatform = Literal["node", "browser", "neutral"]


@dataclass(frozen=True)
class BundleFacetsOptions:
    plugin: FacetBundlePlugin
    entries: Mapping[str, str]
    outdir: str
    working_directory: str | None = None
    external: Sequence[str] | None = None
    source_map: bool | None = None
    minify: bool | None = None
    define: Mapping[str, str] | None = None
    platform: FacetBundlePlatform | None = None
    target: str | Sequence[str] | None = None


@dataclass(frozen=True)
class BundleFacetsResult:
    manifest: FacetBundleManifest
    manifest_path: str


async def bundle_facets(options: BundleFacetsOptions) -> BundleFacetsResult:
    if len(options.plugin.id) == 0:
        raise TypeError("Facet bundle plugin ID must not be empty")
    if isinstance(options.plugin.version, str) and len(options.plugin.version) == 0:
        raise TypeError("Facet bundle plugin version must not be empty")
    if not options.entries:
        raise TypeError("Facet bundle must contain at least one entry")
    for name in _object_keys(options.entries):
        if len(name) == 0:
            raise TypeError("Facet bundle entry name must not be empty")
        if len(options.entries[name]) == 0:
            raise TypeError(f"Facet bundle entry {name} must have a source path")
    for external in options.external if options.external is not None else ():
        if len(external) == 0:
            raise TypeError("Facet bundle external import must not be empty")
    working_directory = os.path.abspath(options.working_directory if options.working_directory is not None else os.getcwd())
    output_directory = os.path.abspath(os.path.join(working_directory, options.outdir))
    parent = os.path.dirname(output_directory)
    await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
    temporary_directory = os.path.join(parent, f".{os.path.basename(output_directory)}.tmp-{uuid.uuid4()}")
    await asyncio.to_thread(os.mkdir, temporary_directory)
    service = EsbuildService()
    try:
        entries: dict[str, FacetBundleEntry] = {}
        collator = Collator.createInstance()
        for entry_name in sorted(_object_keys(options.entries), key=collator.getSortKey):
            entries[entry_name] = await _bundle_entry(
                entry_name, os.path.abspath(os.path.join(working_directory, options.entries[entry_name])),
                temporary_directory, working_directory, options, service,
            )
        manifest = FacetBundleManifest(
            FacetBundlePlugin(options.plugin.id, options.plugin.version),
            MappingProxyType({key: entries[key] for key in _object_keys(entries)}),
        )
        await asyncio.to_thread(Path(temporary_directory, FACET_BUNDLE_MANIFEST_FILE).write_bytes, _manifest_json(manifest))
        await _replace_directory(temporary_directory, output_directory)
        return BundleFacetsResult(manifest, os.path.join(output_directory, FACET_BUNDLE_MANIFEST_FILE))
    except BaseException:
        await asyncio.to_thread(_remove_path, temporary_directory)
        raise
    finally:
        await service.close()


async def _bundle_entry(
    entry_name: str, source: str, temporary_directory: str, working_directory: str,
    options: BundleFacetsOptions, service: EsbuildService,
) -> FacetBundleEntry:
    prefix = "facet-" + hashlib.sha256(_utf8(entry_name)).hexdigest()[:12]
    platform = options.platform if options.platform is not None else "node"
    target = options.target if options.target is not None else "node22.19" if platform == "node" else "es2022"
    targets = [target] if isinstance(target, str) else list(target)
    external = list(dict.fromkeys(["@earendil-works/chord", "@earendil-works/chord/*", *(options.external or ())]))
    try:
        for item in targets:
            if not isinstance(item, str):
                raise EsbuildFailure([{"text": "Expected value for target to be a string", "location": None}], [])
            if "," in item:
                raise EsbuildFailure([{"text": f"Invalid target: {item}", "location": None}], [])
        flags = ["--log-level=silent", "--log-limit=0", "--legal-comments=none"]
        if os.isatty(2):
            flags.insert(0, "--color=true")
        # Empty target strings are omitted by the upstream JavaScript adapter.
        if not isinstance(target, str) or target:
            flags.append("--target=" + ",".join(targets))
        flags.extend(["--format=cjs", f"--platform={platform}"])
        if options.minify:
            flags.append("--minify")
        if options.define is not None:
            for key in _object_keys(options.define):
                value = options.define[key]
                if "=" in key:
                    raise EsbuildFailure([{"text": f"Invalid define: {key}", "location": None}], [])
                if not isinstance(value, str):
                    raise EsbuildFailure([{"text": f"Expected value for define {key!r} to be a string", "location": None}], [])
                flags.append(f"--define:{key}={value}")
        flags.append("--supported:dynamic-import=false")
        if options.source_map is True:
            flags.append("--sourcemap=external")
        flags.extend(["--bundle", "--metafile", f"--outdir={temporary_directory}", f"--entry-names={prefix}-[hash]"])
        flags.extend(f"--external:{item}" for item in external)
        flags.extend(['--banner:js="use strict";', "--out-extension:.js=.cjs"])
        metadata = await service.build(source, flags, working_directory)
    except Exception as error:
        diagnostics: list[str] = []
        for message in getattr(error, "errors", []):
            if not isinstance(message, dict) or not isinstance(message.get("text"), str):
                continue
            location = message.get("location", UNDEFINED)
            if location is None:
                diagnostics.append(message["text"])
            elif isinstance(location, dict):
                diagnostics.append(f"{location['file']}:{location['line']}:{location['column'] + 1}: {message['text']}")
        detail = "\n" + "\n".join(diagnostics) if diagnostics else ""
        raise RuntimeError(f"Could not bundle facet entry {entry_name}{detail}") from error
    if metadata is None:
        raise RuntimeError(f"Facet entry {entry_name} produced no build metadata")
    outputs = cast(dict[str, dict[str, object]], metadata["outputs"])
    matches = [(path, outputs[path]) for path in _object_keys(outputs) if "entryPoint" in outputs[path] and os.path.splitext(path)[1] == ".cjs"]
    if len(matches) != 1:
        raise RuntimeError(f"Facet entry {entry_name} did not produce exactly one JavaScript file")
    output_path, output_metadata = matches[0]
    absolute_output = os.path.abspath(os.path.join(working_directory, output_path))
    file = os.path.relpath(absolute_output, temporary_directory)
    if file == "." or file.startswith(".." + os.sep) or os.path.basename(file) != file:
        raise RuntimeError(f"Facet entry {entry_name} produced an invalid output path")
    source_map: str | Undefined = file + ".map" if options.source_map is True else UNDEFINED
    allowed = {absolute_output}
    if isinstance(source_map, str):
        allowed.add(os.path.join(temporary_directory, source_map))
    if any(os.path.abspath(os.path.join(working_directory, path)) not in allowed for path in outputs):
        raise RuntimeError(f"Facet entry {entry_name} produced files other than its JavaScript bundle and source map")
    contents = await asyncio.to_thread(Path(absolute_output).read_bytes)
    if isinstance(source_map, str):
        await asyncio.to_thread(os.stat, os.path.join(temporary_directory, source_map))
    imports = cast(list[dict[str, object]], output_metadata["imports"])
    external_imports = tuple(sorted(
        {cast(str, item["path"]) for item in imports if item.get("external")},
        key=lambda value: value.encode("utf-16-be", "surrogatepass"),
    ))
    return FacetBundleEntry(file, "sha256-" + base64.b64encode(hashlib.sha256(contents).digest()).decode("ascii"), external_imports, source_map)


async def _replace_directory(temporary_directory: str, output_directory: str) -> None:
    backup = output_directory + ".old-" + str(uuid.uuid4())
    moved_existing = False
    try:
        await asyncio.to_thread(os.rename, output_directory, backup)
        moved_existing = True
    except FileNotFoundError:
        pass
    try:
        await asyncio.to_thread(os.rename, temporary_directory, output_directory)
    except BaseException:
        if moved_existing:
            await asyncio.to_thread(os.rename, backup, output_directory)
        raise
    if moved_existing:
        await asyncio.to_thread(_remove_path, backup)


def _remove_path(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except FileNotFoundError:
        pass


__all__ = ["FacetBundlePlatform", "BundleFacetsOptions", "BundleFacetsResult", "bundle_facets"]
