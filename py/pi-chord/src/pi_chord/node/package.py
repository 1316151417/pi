"""Package metadata and entry resolution from ``chord/src/node/package.ts``."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from .._undefined import UNDEFINED
from .bundle import BundleFacetsOptions, BundleFacetsResult, bundle_facets
from .bundle_loader import _object_keys, _record, _reject_json_constant
from .manifest import FacetBundlePlugin


@dataclass(frozen=True)
class BundleFacetPackageOptions:
    package_path: str
    outdir: str
    default_facets: Mapping[str, str] | None = None


@dataclass(frozen=True)
class BundleFacetPackageResult(BundleFacetsResult):
    package_directory: str
    package_json_path: str


@dataclass(frozen=True)
class _ChordConfiguration:
    facets: Mapping[str, str | Literal[False]]
    external: tuple[str, ...]
    source_map: bool


@dataclass(frozen=True)
class _FacetPackageMetadata:
    package_directory: str
    package_json_path: str
    name: str
    version: str
    peer_dependencies: tuple[str, ...]
    configured_facets: Mapping[str, str | Literal[False]]
    external: tuple[str, ...]
    source_map: bool


async def bundle_facet_package(options: BundleFacetPackageOptions) -> BundleFacetPackageResult:
    metadata = await _read_facet_package_metadata(options.package_path)
    entries = await _resolve_facet_entries(metadata, options.default_facets if options.default_facets is not None else {})
    external = [item for specifier in (*metadata.peer_dependencies, *metadata.external) for item in (specifier, f"{specifier}/*")]
    result = await bundle_facets(BundleFacetsOptions(
        plugin=FacetBundlePlugin(metadata.name, metadata.version), entries=entries, outdir=options.outdir,
        working_directory=metadata.package_directory, external=external, source_map=metadata.source_map,
    ))
    return BundleFacetPackageResult(result.manifest, result.manifest_path, metadata.package_directory, metadata.package_json_path)


async def _read_facet_package_metadata(package_path: str) -> _FacetPackageMetadata:
    if len(package_path) == 0:
        raise TypeError("Facet package path must not be empty")
    candidate = os.path.abspath(package_path)
    try:
        candidate_stats = await asyncio.to_thread(os.stat, candidate)
    except OSError as error:
        raise RuntimeError(f"Could not access facet package {candidate}") from error
    if stat.S_ISDIR(candidate_stats.st_mode):
        package_directory = await asyncio.to_thread(os.path.realpath, candidate, strict=True)
        package_json_path = os.path.join(package_directory, "package.json")
    elif stat.S_ISREG(candidate_stats.st_mode) and os.path.basename(candidate) == "package.json":
        package_json_path = await asyncio.to_thread(os.path.realpath, candidate, strict=True)
        package_directory = os.path.dirname(package_json_path)
    else:
        raise RuntimeError(f"Facet package path must name a directory or package.json: {candidate}")
    try:
        contents = await asyncio.to_thread(Path(package_json_path).read_bytes)
        parsed: object = json.loads(contents.decode("utf-8", "replace"), parse_int=float, parse_constant=_reject_json_constant)
    except Exception as error:
        raise RuntimeError(f"Could not read facet package metadata {package_json_path}") from error
    record = _record(parsed)
    if record is None:
        raise RuntimeError(f"Facet package metadata must be an object: {package_json_path}")
    name, version = record.get("name"), record.get("version")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Facet package must have a non-empty name: {package_json_path}")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"Facet package must have a non-empty version: {package_json_path}")
    peer_dependencies = _parse_peer_dependencies(record.get("peerDependencies", UNDEFINED), package_json_path)
    chord = _parse_chord_configuration(record.get("chord", UNDEFINED), package_json_path)
    return _FacetPackageMetadata(
        package_directory, package_json_path, name, version, peer_dependencies,
        chord.facets, chord.external, chord.source_map,
    )


def _parse_peer_dependencies(value: object, package_json_path: str) -> tuple[str, ...]:
    if value is UNDEFINED:
        return ()
    record = _record(value)
    if record is None or any(not name or not isinstance(version, str) for name, version in record.items()):
        raise RuntimeError(f"Facet package has invalid peerDependencies: {package_json_path}")
    return tuple(sorted(record, key=lambda value: value.encode("utf-16-be", "surrogatepass")))


def _parse_chord_configuration(value: object, package_json_path: str) -> _ChordConfiguration:
    if value is UNDEFINED:
        return _ChordConfiguration(MappingProxyType({}), (), True)
    record = _record(value)
    if record is None:
        raise RuntimeError(f"Facet package chord configuration must be an object: {package_json_path}")
    if any(key not in ("facets", "external", "sourceMap") for key in record):
        raise RuntimeError(f"Facet package chord configuration has an unknown field: {package_json_path}")
    facets: dict[str, str | Literal[False]] = {}
    configured = record.get("facets", UNDEFINED)
    if configured is not UNDEFINED:
        mappings = _record(configured)
        if mappings is None:
            raise RuntimeError(f"Facet package chord.facets must be an object: {package_json_path}")
        for name in _object_keys(mappings):
            source = mappings[name]
            if not name or (not isinstance(source, str) and source is not False) or source == "":
                raise RuntimeError(f"Facet package has an invalid chord.facets entry: {package_json_path}")
            facets[name] = cast(str | Literal[False], source)
    external: tuple[str, ...] = ()
    configured_external = record.get("external", UNDEFINED)
    if configured_external is not UNDEFINED:
        if not isinstance(configured_external, list) or any(not isinstance(item, str) or not item for item in configured_external):
            raise RuntimeError(f"Facet package chord.external must contain non-empty strings: {package_json_path}")
        external = tuple(sorted(set(cast(list[str], configured_external)), key=lambda value: value.encode("utf-16-be", "surrogatepass")))
    source_map = record.get("sourceMap", UNDEFINED)
    if source_map is not UNDEFINED and not isinstance(source_map, bool):
        raise RuntimeError(f"Facet package chord.sourceMap must be a boolean: {package_json_path}")
    return _ChordConfiguration(MappingProxyType(facets), external, True if source_map is UNDEFINED else cast(bool, source_map))


async def _resolve_facet_entries(metadata: _FacetPackageMetadata, default_facets: Mapping[str, str]) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    for name in _object_keys(default_facets):
        source = default_facets[name]
        _validate_facet_mapping(name, source, "default")
        path = _resolve_package_entry(metadata.package_directory, source, name)
        try:
            entry_stats = await asyncio.to_thread(os.stat, path)
            if not stat.S_ISREG(entry_stats.st_mode):
                raise RuntimeError(f"Default facet entry {name} is not a file: {path}")
            canonical_path = await asyncio.to_thread(os.path.realpath, path, strict=True)
            _validate_canonical_package_entry(metadata.package_directory, canonical_path, name)
            entries[name] = canonical_path
        except FileNotFoundError:
            continue
    for name in _object_keys(metadata.configured_facets):
        source = metadata.configured_facets[name]
        if source is False:
            entries.pop(name, None)
            continue
        _validate_facet_mapping(name, source, "configured")
        path = _resolve_package_entry(metadata.package_directory, source, name)
        try:
            entry_stats = await asyncio.to_thread(os.stat, path)
        except OSError as error:
            raise RuntimeError(f"Could not access configured facet entry {name}: {path}") from error
        if not stat.S_ISREG(entry_stats.st_mode):
            raise RuntimeError(f"Configured facet entry {name} is not a file: {path}")
        canonical_path = await asyncio.to_thread(os.path.realpath, path, strict=True)
        _validate_canonical_package_entry(metadata.package_directory, canonical_path, name)
        entries[name] = canonical_path
    if not entries:
        raise RuntimeError(f"Facet package {metadata.name} has no configured or conventional facet entries")
    return MappingProxyType({name: entries[name] for name in _object_keys(entries)})


def _validate_facet_mapping(name: str, source: str, kind: str) -> None:
    if not name:
        raise RuntimeError(f"Facet package {kind} entry name must not be empty")
    if not source:
        raise RuntimeError(f"Facet package {kind} entry {name} must have a source path")


def _resolve_package_entry(package_directory: str, source: str, name: str) -> str:
    if os.path.isabs(source):
        raise RuntimeError(f"Facet package entry {name} must be relative to the package directory")
    path = os.path.abspath(os.path.join(package_directory, source))
    try:
        relative = os.path.relpath(path, package_directory)
    except ValueError:
        relative = path
    if relative in (".", "..") or relative.startswith(".." + os.sep) or os.path.isabs(relative):
        raise RuntimeError(f"Facet package entry {name} escapes the package directory")
    return path


def _validate_canonical_package_entry(package_directory: str, path: str, name: str) -> None:
    try:
        relative = os.path.relpath(path, package_directory)
    except ValueError:
        relative = path
    if relative == ".." or relative.startswith(".." + os.sep) or os.path.isabs(relative):
        raise RuntimeError(f"Facet package entry {name} resolves outside the package directory")


__all__ = ["BundleFacetPackageOptions", "BundleFacetPackageResult", "bundle_facet_package"]
