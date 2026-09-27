"""Eager, reusable Promise adapters whose waiters do not cancel shared work."""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Generator
from typing import cast


class Promise[T](Awaitable[T]):
    def __init__(self, future: asyncio.Future[T]) -> None:
        self.future = future

    def __await__(self) -> Generator[object, None, T]:
        # A JavaScript await schedules a continuation even for an already settled promise.
        yield from asyncio.sleep(0).__await__()
        return (yield from asyncio.shield(self.future).__await__())

    def then_settled(self, callback: Callable[[], object]) -> None:
        def settled(future: asyncio.Future[T]) -> None:
            if not future.cancelled():
                future.exception()
            callback()
        self.future.add_done_callback(settled)


def resolved[T](value: T) -> Promise[T]:
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return Promise(future)


def rejected(error: BaseException) -> Promise:
    future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    future.set_exception(error)
    return Promise(future)


def spawn[T](coroutine: Coroutine[object, object, T]) -> Promise[T]:
    return Promise(asyncio.get_running_loop().create_task(coroutine))


def promise[T](value: Awaitable[T]) -> Promise[T]:
    if isinstance(value, Promise):
        return value
    return Promise(asyncio.ensure_future(value))


def capture[T](callback: Callable[[], T]) -> Promise[T]:
    try:
        return resolved(callback())
    except BaseException as error:
        return cast(Promise[T], rejected(error))


async def settled_all(values: list[Awaitable[object]]) -> list[object]:
    return list(await asyncio.gather(*values, return_exceptions=True))
