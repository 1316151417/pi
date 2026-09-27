"""Shallow model-catalogue flattening from ``model-catalog.ts``."""

from collections.abc import Mapping

from .utils._javascript import javascript_object_keys

type ModelGroups = Mapping[str, Mapping[str, object]]
type ModelCatalog[TModel] = dict[str, TModel]


def flatten_model_catalog[TModel](_provider: str, groups: Mapping[str, Mapping[str, TModel]]) -> ModelCatalog[TModel]:
    result: dict[str, TModel] = {}
    for api in javascript_object_keys(groups):
        group = groups[api]
        for id in javascript_object_keys(group):
            result[id] = group[id]
    return {id: result[id] for id in javascript_object_keys(result)}


__all__ = ["ModelGroups", "ModelCatalog", "flatten_model_catalog"]
