"""Public Chord composition entry points from src/api.ts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .facets.host import FacetKernel
from .facets.loader import dispose_loaded_facets
from .services.consumer import RemoteServiceBindingImpl
from .services.provider import RemoteServiceProvider
from .services.state import MutableReplicatedStateImpl
from .types import Facet, FacetLoader, FacetOptions, LoadedFacets, RemoteServiceBindingOptions, Service


@dataclass(frozen=True)
class _FacetHost:
    _kernel: FacetKernel

    @property
    def services(self) -> RemoteServiceProvider:
        return self._kernel.provider

    async def reload(self, facets: Sequence[Facet]) -> None:
        await self._kernel.reload(facets)

    async def dispose(self) -> None:
        await self._kernel.dispose()


async def create_facet_host(options: FacetOptions) -> _FacetHost:
    kernel = FacetKernel(options)
    await kernel.activate()
    return _FacetHost(kernel)


@dataclass(frozen=True)
class _StaticLoadedFacets:
    facets: tuple[Facet, ...]

    async def dispose(self) -> None:
        pass


@dataclass(frozen=True)
class _StaticFacetLoader:
    facets: tuple[Facet, ...]

    async def load(self) -> _StaticLoadedFacets:
        return _StaticLoadedFacets(self.facets)


def create_static_facet_loader(facets: Sequence[Facet]) -> FacetLoader:
    return _StaticFacetLoader(tuple(facets))


class _CombinedLoadedFacets:
    def __init__(self, loaded: list[LoadedFacets]) -> None:
        self.facets = tuple(facet for entry in loaded for facet in entry.facets)
        self._loaded = loaded
        self._disposed = False

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors = await dispose_loaded_facets(list(reversed(self._loaded)))
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to dispose loaded facets", errors)


@dataclass(frozen=True)
class _CombinedFacetLoader:
    loaders: Sequence[FacetLoader]

    async def load(self) -> LoadedFacets:
        loaded: list[LoadedFacets] = []
        try:
            for loader in self.loaders:
                loaded.append(await loader.load())
        except BaseException as error:
            cleanup_errors = await dispose_loaded_facets(list(reversed(loaded)))
            if cleanup_errors:
                raise BaseExceptionGroup("Facet loading and cleanup failed", [error, *cleanup_errors])
            raise
        return _CombinedLoadedFacets(loaded)


def combine_facet_loaders(loaders: Sequence[FacetLoader]) -> FacetLoader:
    return _CombinedFacetLoader(loaders)


def define_facet(facet: Facet) -> Facet:
    return facet


def define_service[T](id: str, *, local: bool = False) -> Service[T]:
    if not id:
        raise TypeError("Service ID must not be empty")
    if id.startswith("$chord."):
        raise TypeError("Service IDs beginning with $chord. are reserved")
    return Service(id, local)


def create_remote_service_binding(options: RemoteServiceBindingOptions) -> RemoteServiceBindingImpl:
    return RemoteServiceBindingImpl(options)


def replicated_state[T](initial: T) -> MutableReplicatedStateImpl[T]:
    return MutableReplicatedStateImpl(initial)
