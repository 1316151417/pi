"""Concurrent loader cleanup from facets/loader.ts."""

import asyncio
from collections.abc import Sequence

from ..types import LoadedFacets


async def dispose_loaded_facets(loaded: Sequence[LoadedFacets]) -> list[BaseException]:
    results = await asyncio.gather(*(entry.dispose() for entry in loaded), return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]
