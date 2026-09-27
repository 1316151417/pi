"""Load the bundled, immutable-source JSON snapshots into mutable model views."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

from ..model_catalog import flatten_model_catalog
from ..types import Model, model_from_json


def load_model_catalog(provider: str) -> dict[str, Model]:
    resource = files("pi_ai.providers").joinpath("data", provider + ".json")
    groups = cast(dict[str, dict[str, dict[str, object]]], json.loads(resource.read_text(encoding="utf-8")))
    return flatten_model_catalog(provider, {
        api: {model_id: model_from_json(value) for model_id, value in models.items()}
        for api, models in groups.items()
    })
