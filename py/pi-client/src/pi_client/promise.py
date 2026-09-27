"""Reusable resolver records corresponding to ``promise.ts``."""

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PromiseResolvers(Generic[T]):
    promise: asyncio.Future[T]

    def resolve(self, value: T) -> None:
        if not self.promise.done():
            self.promise.set_result(value)

    def reject(self, reason: BaseException) -> None:
        if not self.promise.done():
            self.promise.set_exception(reason)


def create_promise_resolvers[T]() -> PromiseResolvers[T]:
    return PromiseResolvers(asyncio.get_running_loop().create_future())
