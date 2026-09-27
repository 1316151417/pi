"""Bundle data and loader lifecycle from ``node/bundle-loader.ts``.

Both factories require a synchronous ``execute_common_js`` adapter. It owns the
original ``executeCommonJsModule`` boundary: JavaScript evaluation, controlled
Node module resolution, and bridging exported functions/objects into Python.
There is no default CommonJS interpreter, and version 2 artifacts remain JS.

The adapter returns module exports as a mapping or an object with a ``default``
attribute. Its exported facets must be ``Facet`` instances or bridge objects
with ``id`` and callable ``setup`` attributes; their identities are preserved.
String paths remain filesystem paths; pass a urllib ParseResult/SplitResult for
the TypeScript URL overload. Reads preserve newlines and replace invalid UTF-8.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast
from urllib.parse import ParseResult, SplitResult, unquote

from .._undefined import UNDEFINED, Undefined
from ..types import Facet, FacetLoader, LoadedFacets
from .manifest import (
    FACET_BUNDLE_ARTIFACT_FORMAT, FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
    FACET_BUNDLE_FORMAT, FACET_BUNDLE_FORMAT_VERSION, FACET_BUNDLE_MANIFEST_FILE,
    FacetBundleArtifact, FacetBundleEntry, FacetBundleManifest, FacetBundlePlugin,
)

type FilePath = str | os.PathLike[str] | ParseResult | SplitResult
type FacetBundleExternalResolver = Callable[[str], str | ParseResult | SplitResult | Undefined]


class CommonJsExecutor(Protocol):
    """Host-supplied implementation of TS ``executeCommonJsModule``.

    Each call must execute a fresh entry without caching its module, while host
    external modules retain their normal cache. Before evaluation, resolve all
    declared externals, applying ``resolve_external`` first (UNDEFINED delegates
    to the host's defaults). Preserve the source's package-specifier validation,
    built-in handling, Chord export mappings, and file/package/node URL rules.
    Both ``require`` and ``require.resolve`` must reject undeclared imports.

    The JS wrapper receives ``exports``, ``require``, ``module``, ``__filename``
    and ``__dirname``; its initial ``this`` is the null-prototype exports object.
    Return the final ``module.exports`` after synchronous evaluation, bridging
    its facet objects and setup callbacks into Python. Keep runtime references
    alive through the returned objects, rather than through a loader-global
    entry cache. This callable owns the complete execution/resolution boundary;
    merely parsing exports or returning static facets is not an implementation
    of that boundary.
    """

    def __call__(
        self, source: str, module_path: str, external_imports: Sequence[str],
        resolve_external: FacetBundleExternalResolver | None, /,
    ) -> object: ...


@dataclass(frozen=True)
class FacetBundleLoaderOptions:
    manifest_path: FilePath
    entry: str
    verify_integrity: bool = True
    resolve_external: FacetBundleExternalResolver | None = None


@dataclass(frozen=True)
class FacetBundleArtifactLoaderOptions:
    artifact: object
    resolve_external: FacetBundleExternalResolver | None = None
    temporary_directory: str | None = None


@dataclass(frozen=True)
class ReadFacetBundleArtifactOptions:
    manifest_path: FilePath
    entry: str


def create_facet_bundle_loader(
    options: FacetBundleLoaderOptions, *, execute_common_js: CommonJsExecutor,
) -> FacetLoader:
    """Create a reusable loader; each load evaluates a fresh CommonJS entry."""
    if len(options.entry) == 0:
        raise TypeError("Facet bundle entry name must not be empty")
    manifest_path = _to_file_path(options.manifest_path)
    if not callable(execute_common_js):
        raise TypeError("execute_common_js must be a synchronous CommonJS execution adapter")
    return _BundleLoader(manifest_path, options, execute_common_js)


def create_facet_bundle_artifact_loader(
    options: FacetBundleArtifactLoaderOptions, *, execute_common_js: CommonJsExecutor,
) -> FacetLoader:
    """Verify an artifact now and materialize a separate generation per load."""
    artifact = _validate_artifact(options.artifact)
    temporary_parent = os.path.abspath(
        tempfile.gettempdir() if options.temporary_directory is None else options.temporary_directory,
    )
    if not callable(execute_common_js):
        raise TypeError("execute_common_js must be a synchronous CommonJS execution adapter")
    return _ArtifactLoader(artifact, temporary_parent, options.resolve_external, execute_common_js)


class _BundleLoader:
    def __init__(
        self, manifest_path: str, options: FacetBundleLoaderOptions, execute_common_js: CommonJsExecutor,
    ) -> None:
        self._manifest_path = manifest_path
        self._options = options
        self._execute_common_js = execute_common_js

    async def load(self) -> LoadedFacets:
        manifest = await read_facet_bundle_manifest(self._manifest_path)
        entry = manifest.entries.get(self._options.entry)
        if entry is None:
            raise RuntimeError(f"Facet bundle {manifest.plugin.id} has no entry named {self._options.entry}")
        module_path = _resolve_bundle_file(self._manifest_path, entry.file, "entry")
        try:
            contents = await asyncio.to_thread(Path(module_path).read_bytes)
            source = contents.decode("utf-8", errors="replace")
            if self._options.verify_integrity is not False:
                _verify_source(source, entry)
            exported = self._execute_common_js(
                source, module_path, entry.external_imports, self._options.resolve_external,
            )
            facets = _facets_from_module(exported, manifest.plugin.id, self._options.entry)
            return _LoadedBundleFacets(facets)
        except Exception as error:
            raise RuntimeError(
                f"Could not load facet bundle entry {manifest.plugin.id}/{self._options.entry}: {error}",
            ) from error


class _LoadedBundleFacets:
    def __init__(self, facets: tuple[Facet, ...]) -> None:
        self._facets = facets
        self._disposed = False

    @property
    def facets(self) -> tuple[Facet, ...]:
        return self._facets

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._facets = ()


class _ArtifactLoader:
    def __init__(
        self, artifact: FacetBundleArtifact, temporary_parent: str,
        resolve_external: FacetBundleExternalResolver | None, execute_common_js: CommonJsExecutor,
    ) -> None:
        self._artifact = artifact
        self._temporary_parent = temporary_parent
        self._resolve_external = resolve_external
        self._execute_common_js = execute_common_js

    async def load(self) -> LoadedFacets:
        await asyncio.to_thread(os.makedirs, self._temporary_parent, exist_ok=True)
        # Do not suspend between allocating a directory and installing cleanup.
        directory = tempfile.mkdtemp(prefix="chord-facet-", dir=self._temporary_parent)
        try:
            await _materialize_artifact(directory, self._artifact)
            loaded = await create_facet_bundle_loader(
                FacetBundleLoaderOptions(
                    manifest_path=os.path.join(directory, FACET_BUNDLE_MANIFEST_FILE),
                    entry=self._artifact.entry_name,
                    resolve_external=self._resolve_external,
                ),
                execute_common_js=self._execute_common_js,
            ).load()
            return _LoadedArtifactFacets(loaded, directory)
        except (Exception, asyncio.CancelledError) as error:
            try:
                await asyncio.to_thread(_remove_directory, directory)
            except (Exception, asyncio.CancelledError) as cleanup_error:
                raise BaseExceptionGroup(
                    "Facet bundle artifact loading and cleanup failed", [error, cleanup_error],
                ) from None
            raise


class _LoadedArtifactFacets:
    def __init__(self, loaded: LoadedFacets, directory: str) -> None:
        self._loaded = loaded
        self._directory = directory
        self._disposed = False

    @property
    def facets(self) -> Sequence[Facet]:
        return self._loaded.facets

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors: list[BaseException] = []
        try:
            await self._loaded.dispose()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        try:
            await asyncio.to_thread(_remove_directory, self._directory)
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to dispose facet bundle artifact", errors)


def _remove_directory(directory: str) -> None:
    try:
        if os.path.isdir(directory) and not os.path.islink(directory):
            shutil.rmtree(directory)
        else:
            os.unlink(directory)
    except FileNotFoundError:
        pass


def _is_export_object(value: object) -> bool:
    return not isinstance(value, (
        type(None), Undefined, bool, int, float, complex, str, bytes,
        bytearray, memoryview, list, tuple, set, frozenset,
    )) and not callable(value)


def _facets_from_module(imported: object, plugin_id: str, entry_name: str) -> tuple[Facet, ...]:
    if not _is_export_object(imported):
        raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} did not export a module")
    exported = (
        imported.get("default", UNDEFINED) if isinstance(imported, Mapping)
        else getattr(imported, "default", UNDEFINED)
    )
    candidates = exported if isinstance(exported, (list, tuple)) else (exported,)
    if not candidates:
        raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} exported no facets")
    facets: list[Facet] = []
    for candidate in candidates:
        if not _is_export_object(candidate):
            raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} has a facet with an invalid ID")
        facet_id: object = getattr(candidate, "id", UNDEFINED)
        if not isinstance(facet_id, str) or not facet_id:
            raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} has a facet with an invalid ID")
        if not callable(getattr(candidate, "setup", UNDEFINED)):
            raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} facet {facet_id} has no setup function")
        facets.append(cast(Facet, candidate))
    ids = [facet.id for facet in facets]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Facet bundle entry {plugin_id}/{entry_name} exports duplicate facet IDs")
    return tuple(facets)


async def read_facet_bundle_manifest(path: FilePath) -> FacetBundleManifest:
    manifest_path = _to_file_path(path)
    try:
        source = await asyncio.to_thread(Path(manifest_path).read_bytes)
        parsed: object = json.loads(
            source.decode("utf-8", errors="replace"),
            parse_int=float, parse_constant=_reject_json_constant,
        )
    except Exception as error:
        raise RuntimeError(f"Could not read facet bundle manifest {manifest_path}") from error
    return _validate_manifest(parsed, manifest_path)


async def read_facet_bundle_artifact(options: ReadFacetBundleArtifactOptions) -> FacetBundleArtifact:
    if len(options.entry) == 0:
        raise TypeError("Facet bundle entry name must not be empty")
    manifest_path = _to_file_path(options.manifest_path)
    manifest = await read_facet_bundle_manifest(manifest_path)
    entry = manifest.entries.get(options.entry)
    if entry is None:
        raise RuntimeError(f"Facet bundle {manifest.plugin.id} has no entry named {options.entry}")
    module_path = _resolve_bundle_file(manifest_path, entry.file, "entry")
    contents = await asyncio.to_thread(Path(module_path).read_bytes)
    source = contents.decode("utf-8", errors="replace")
    _verify_source(source, entry)
    source_map_contents: str | Undefined = UNDEFINED
    if isinstance(entry.source_map, str):
        map_path = _resolve_bundle_file(manifest_path, entry.source_map, "source map")
        contents = await asyncio.to_thread(Path(map_path).read_bytes)
        source_map_contents = contents.decode("utf-8", errors="replace")
    return FacetBundleArtifact(
        manifest.plugin, options.entry, entry, source, source_map_contents,
    )


def _to_file_path(path: FilePath) -> str:
    if isinstance(path, (str, os.PathLike)):
        return os.path.abspath(path)
    if path.scheme != "file":
        raise TypeError(f"Facet bundle manifest must be a file URL, not {path.scheme}:")
    if re.search(r"%2f", path.path, re.IGNORECASE):
        raise ValueError("File URL path must not include encoded / characters")
    decoded = unquote(path.path, encoding="utf-8", errors="strict")
    if os.name == "nt":
        if re.search(r"%5c", path.path, re.IGNORECASE):
            raise ValueError("File URL path must not include encoded \\ characters")
        decoded = decoded.replace("/", "\\")
        if path.hostname and path.hostname != "localhost":
            return f"\\\\{path.hostname}{decoded}"
        if not re.match(r"^\\[A-Za-z]:\\", decoded):
            raise ValueError("File URL path must be absolute")
        return decoded[1:]
    if path.hostname and path.hostname != "localhost":
        raise ValueError(f"File URL host must be localhost or empty: {path.hostname}")
    return decoded


def _resolve_bundle_file(manifest_path: str, file: str, label: str) -> str:
    if not file or os.path.isabs(file) or os.path.basename(file) != file or file in (".", ".."):
        raise RuntimeError(f"Facet bundle {label} must be a filename relative to its manifest")
    return os.path.abspath(os.path.join(os.path.dirname(manifest_path), file))


def _utf8(value: str) -> bytes:
    # Node UTF-8 encoding combines surrogate pairs and replaces lone surrogates.
    return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode("utf-8")


def _verify_source(source: str, entry: FacetBundleEntry) -> None:
    expected = _parse_integrity(entry.integrity)
    actual = base64.b64encode(hashlib.sha256(_utf8(source)).digest()).decode("ascii")
    if actual != expected:
        raise RuntimeError(f"Facet bundle integrity check failed for {entry.file}")


def _parse_integrity(integrity: str) -> str:
    prefix = "sha256-"
    if not integrity.startswith(prefix) or len(integrity) == len(prefix):
        raise RuntimeError("Facet bundle entry has an invalid SHA-256 integrity value")
    return integrity[len(prefix):]


def _validate_artifact(value: object) -> FacetBundleArtifact:
    if isinstance(value, FacetBundleArtifact):
        value = value.to_wire()
    record = _record(value)
    if (
        record is None or record.get("format") != FACET_BUNDLE_ARTIFACT_FORMAT
        or type(record.get("formatVersion")) not in (int, float)
        or record.get("formatVersion") != FACET_BUNDLE_ARTIFACT_FORMAT_VERSION
        or not isinstance(record.get("entryName"), str) or not record["entryName"]
        or not isinstance(record.get("source"), str)
    ):
        raise RuntimeError("Invalid facet bundle artifact")
    entry_name = cast(str, record["entryName"])
    source = cast(str, record["source"])
    manifest = _validate_manifest({
        "format": FACET_BUNDLE_FORMAT, "formatVersion": FACET_BUNDLE_FORMAT_VERSION,
        "plugin": record.get("plugin", UNDEFINED),
        "entries": {entry_name: record.get("entry", UNDEFINED)},
    }, "facet bundle artifact")
    entry = manifest.entries[entry_name]
    map_contents = record.get("sourceMapContents", UNDEFINED)
    if entry.source_map is UNDEFINED:
        if map_contents is not UNDEFINED:
            raise RuntimeError("Facet bundle artifact has source map contents without a source map")
    elif not isinstance(map_contents, str):
        raise RuntimeError("Facet bundle artifact is missing its source map contents")
    _verify_source(source, entry)
    return FacetBundleArtifact(
        manifest.plugin, entry_name, entry, source, cast(str | Undefined, map_contents),
    )


async def _materialize_artifact(directory: str, artifact: FacetBundleArtifact) -> None:
    manifest = FacetBundleManifest(artifact.plugin, {artifact.entry_name: artifact.entry})
    writes = [
        asyncio.to_thread(Path(directory, artifact.entry.file).write_bytes, _utf8(artifact.source)),
        asyncio.to_thread(Path(directory, FACET_BUNDLE_MANIFEST_FILE).write_bytes, _manifest_json(manifest)),
    ]
    if isinstance(artifact.entry.source_map, str):
        writes.append(asyncio.to_thread(
            Path(directory, artifact.entry.source_map).write_bytes,
            _utf8(cast(str, artifact.source_map_contents)),
        ))
    await asyncio.gather(*writes)


def _manifest_json(manifest: FacetBundleManifest) -> bytes:
    value = manifest.to_wire()
    entries = cast(dict[str, object], value["entries"])
    value["entries"] = {key: entries[key] for key in _object_keys(entries)}
    # JS JSON.stringify emits Unicode directly except isolated surrogate units.
    source = json.dumps(value, ensure_ascii=False, indent=2)
    source = source.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    source = re.sub("[\ud800-\udfff]", lambda match: f"\\u{ord(match[0]):04x}", source) + "\n"
    return _utf8(source)


def _validate_manifest(value: object, path: str) -> FacetBundleManifest:
    record = _record(value)
    if record is None or record.get("format") != FACET_BUNDLE_FORMAT:
        raise RuntimeError(f"Invalid facet bundle manifest format in {path}")
    version = record.get("formatVersion", UNDEFINED)
    if type(version) not in (int, float) or version != FACET_BUNDLE_FORMAT_VERSION:
        raise RuntimeError(f"Unsupported facet bundle manifest version in {path}: {_js_string(version)}")
    plugin = _record(record.get("plugin"))
    if plugin is None or not isinstance(plugin.get("id"), str) or not plugin["id"]:
        raise RuntimeError(f"Facet bundle manifest has an invalid plugin identity in {path}")
    plugin_version = plugin.get("version", UNDEFINED)
    if plugin_version is not UNDEFINED and (not isinstance(plugin_version, str) or not plugin_version):
        raise RuntimeError(f"Facet bundle manifest has an invalid plugin version in {path}")
    candidates = _record(record.get("entries"))
    if candidates is None or not candidates:
        raise RuntimeError(f"Facet bundle manifest has no entries in {path}")
    entries: dict[str, FacetBundleEntry] = {}
    for name in _object_keys(candidates):
        candidate = _record(candidates[name])
        if not name or candidate is None:
            raise RuntimeError(f"Facet bundle manifest has an invalid entry in {path}")
        file = candidate.get("file")
        if not isinstance(file, str):
            raise RuntimeError(f"Facet bundle entry {name} has no file")
        _resolve_bundle_file(path, file, f"entry {name}")
        integrity = candidate.get("integrity")
        if not isinstance(integrity, str):
            raise RuntimeError(f"Facet bundle entry {name} has no integrity")
        _parse_integrity(integrity)
        imports = candidate.get("externalImports")
        if not isinstance(imports, list) or any(not isinstance(item, str) for item in imports):
            raise RuntimeError(f"Facet bundle entry {name} has invalid external imports")
        external_imports = tuple(cast(list[str], imports))
        if len(set(external_imports)) != len(external_imports):
            raise RuntimeError(f"Facet bundle entry {name} has duplicate external imports")
        source_map = candidate.get("sourceMap", UNDEFINED)
        if source_map is not UNDEFINED:
            if not isinstance(source_map, str):
                raise RuntimeError(f"Facet bundle entry {name} has an invalid source map")
            _resolve_bundle_file(path, source_map, f"entry {name} source map")
        entries[name] = FacetBundleEntry(file, integrity, external_imports, cast(str | Undefined, source_map))
    return FacetBundleManifest(
        FacetBundlePlugin(cast(str, plugin["id"]), cast(str | Undefined, plugin_version)),
        MappingProxyType(entries),
    )


def _record(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return cast(Mapping[str, object], value)
    return None


def _object_keys(record: Mapping[str, object]) -> list[str]:
    indexes: list[tuple[int, str]] = []
    other: list[str] = []
    for key in record:
        if (
            key.isascii() and key.isdigit() and len(key) <= 10
            and (key == "0" or key[0] != "0") and int(key) < 0xFFFFFFFF
        ):
            indexes.append((int(key), key))
        else:
            other.append(key)
    return [key for _, key in sorted(indexes)] + other


def _js_string(value: object) -> str:
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join("" if item is None or item is UNDEFINED else _js_string(item) for item in value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return "NaN"
        if math.isinf(numeric):
            return "Infinity" if numeric > 0 else "-Infinity"
        if numeric == 0:
            return "0"
        raw = repr(numeric)
        if "e" not in raw:
            return raw.removesuffix(".0")
        mantissa, exponent = raw.split("e")
        exp = int(exponent)
        if -6 <= exp < 21:
            sign = "-" if mantissa.startswith("-") else ""
            digits = mantissa.lstrip("-").replace(".", "").rstrip("0")
            point = exp + 1
            if point <= 0:
                return sign + "0." + "0" * -point + digits
            if point >= len(digits):
                return sign + digits + "0" * (point - len(digits))
            return sign + digits[:point] + "." + digits[point:]
        return mantissa.removesuffix(".0") + "e" + ("+" if exp >= 0 else "-") + str(abs(exp))
    return "[object Object]"


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Invalid JSON constant: {value}")


__all__ = [
    "FilePath", "FacetBundleExternalResolver", "CommonJsExecutor", "FacetBundleLoaderOptions",
    "FacetBundleArtifactLoaderOptions", "ReadFacetBundleArtifactOptions",
    "create_facet_bundle_loader", "create_facet_bundle_artifact_loader",
    "read_facet_bundle_manifest", "read_facet_bundle_artifact",
]
