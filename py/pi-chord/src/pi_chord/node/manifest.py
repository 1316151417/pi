"""Versioned bundle data contracts from ``chord/src/node/manifest.ts``.

Python fields use snake_case; ``to_wire`` emits the original JSON field names.
Optional fields retain UNDEFINED so an explicitly supplied null stays distinct.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .._undefined import UNDEFINED, Undefined

FACET_BUNDLE_FORMAT = "chord.facet-bundle"
FACET_BUNDLE_FORMAT_VERSION = 2
FACET_BUNDLE_MANIFEST_FILE = "chord-facets.json"
FACET_BUNDLE_ARTIFACT_FORMAT = "chord.facet-bundle-artifact"
FACET_BUNDLE_ARTIFACT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class FacetBundleEntry:
    file: str
    integrity: str
    external_imports: tuple[str, ...]
    source_map: str | Undefined = UNDEFINED

    def to_wire(self) -> dict[str, object]:
        result: dict[str, object] = {
            "file": self.file, "integrity": self.integrity,
            "externalImports": list(self.external_imports),
        }
        if self.source_map is not UNDEFINED:
            result["sourceMap"] = self.source_map
        return result


@dataclass(frozen=True)
class FacetBundlePlugin:
    id: str
    version: str | Undefined = UNDEFINED

    def to_wire(self) -> dict[str, object]:
        result: dict[str, object] = {"id": self.id}
        if self.version is not UNDEFINED:
            result["version"] = self.version
        return result


@dataclass(frozen=True)
class FacetBundleManifest:
    plugin: FacetBundlePlugin
    entries: Mapping[str, FacetBundleEntry]
    format: Literal["chord.facet-bundle"] = FACET_BUNDLE_FORMAT
    format_version: Literal[2] = FACET_BUNDLE_FORMAT_VERSION

    def to_wire(self) -> dict[str, object]:
        return {
            "format": self.format, "formatVersion": self.format_version,
            "plugin": self.plugin.to_wire(),
            "entries": {name: entry.to_wire() for name, entry in self.entries.items()},
        }


@dataclass(frozen=True)
class FacetBundleArtifact:
    plugin: FacetBundlePlugin
    entry_name: str
    entry: FacetBundleEntry
    source: str
    source_map_contents: str | Undefined = UNDEFINED
    format: Literal["chord.facet-bundle-artifact"] = FACET_BUNDLE_ARTIFACT_FORMAT
    format_version: Literal[2] = FACET_BUNDLE_ARTIFACT_FORMAT_VERSION

    def to_wire(self) -> dict[str, object]:
        result: dict[str, object] = {
            "format": self.format, "formatVersion": self.format_version,
            "plugin": self.plugin.to_wire(), "entryName": self.entry_name,
            "entry": self.entry.to_wire(), "source": self.source,
        }
        if self.source_map_contents is not UNDEFINED:
            result["sourceMapContents"] = self.source_map_contents
        return result


__all__ = [
    "FACET_BUNDLE_FORMAT", "FACET_BUNDLE_FORMAT_VERSION", "FACET_BUNDLE_MANIFEST_FILE",
    "FACET_BUNDLE_ARTIFACT_FORMAT", "FACET_BUNDLE_ARTIFACT_FORMAT_VERSION",
    "FacetBundleEntry", "FacetBundlePlugin", "FacetBundleManifest", "FacetBundleArtifact",
]
