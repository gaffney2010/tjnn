from typing import Any

# TJ: Use generics
Elem = Any


# Will only push a new element if it never existed in the queue
class UniqueQueue(object):
    def __init__(self):
        self._queue = list()
        self._seen = set()

    def push(self, x: Elem) -> None:
        if x in self._seen:
            return
        self._queue.append(x)
        self._seen.add(x)

    def pop(self) -> Elem:
        result = self._queue[0]
        self._queue = self._queue[1:]
        return result

    def __bool__(self) -> bool:
        return len(self._queue) > 0
