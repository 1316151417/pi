"""Image catalog copied from the tracked image-models.generated.ts.

Source SHA-256: eca804167d721a157c997bc22d76e6f869da35467782eb4b0de1f6b3b2d8ff66
"""

import json
from importlib.resources import files
from typing import cast

from .types import ImagesModel, images_model_from_json

_values = cast(dict[str, dict[str, dict[str, object]]], json.loads(
    files("pi_ai").joinpath("image_models_data.json").read_text(encoding="utf-8"),
))
IMAGE_MODELS: dict[str, dict[str, ImagesModel]] = {
    provider: {model_id: images_model_from_json(value) for model_id, value in models.items()}
    for provider, models in _values.items()
}
