"""Insertion-ordered listener sets with JavaScript Set's live iteration."""

from collections.abc import Hashable, Iterator, MutableMapping


class LiveMap[K: Hashable, V](MutableMapping[K, V]):
    """Map iteration sees later insertions, including delete-and-reinsert.

    Monotonic positions avoid retaining deleted values or invalidating Python
    dict iterators while user callbacks mutate the collection.
    """

    def __init__(self) -> None:
        self._items: dict[K, tuple[V, int]] = {}
        self._next = 0

    def __getitem__(self, key: K) -> V:
        return self._items[key][0]

    def __setitem__(self, key: K, value: V) -> None:
        previous = self._items.get(key)
        if previous is not None:
            self._items[key] = (value, previous[1])
        else:
            self._items[key] = (value, self._next)
            self._next += 1

    def __delitem__(self, key: K) -> None:
        del self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[K]:
        position = -1
        while True:
            item = next(((key, slot[1]) for key, slot in self._items.items() if slot[1] > position), None)
            if item is None:
                return
            key, position = item
            yield key

    def clear(self) -> None:
        self._items.clear()


class ListenerSet[T]:
    def __init__(self) -> None:
        self._items: dict[int, tuple[T, int]] = {}
        self._next = 0

    def add(self, listener: T) -> None:
        if id(listener) not in self._items:
            self._items[id(listener)] = (listener, self._next)
            self._next += 1

    def discard(self, listener: T) -> None:
        self._items.pop(id(listener), None)

    def __iter__(self) -> Iterator[T]:
        position = -1
        while True:
            next_item = next((item for item in self._items.values() if item[1] > position), None)
            if next_item is None:
                return
            listener, position = next_item
            yield listener
