from dataclasses import dataclass, field
from typing import Self, Iterable


#fixme this is dumb, it should be replacable with something from sdk
@dataclass
class _stack[T]:
    head: T = field(default=None)
    tail: list[T] = field(default_factory=list)

    def push(self, x: T) -> Self:
        if self.head is not None:
            self.tail.append(self.head)
        self.head = x
        return self

    def pop(self) -> T:
        assert self.head is not None #todo exception
        out = self.head
        self.head = self.tail[-1] if self.tail else None
        self.tail = self.tail[:-1] if self.tail else self.tail
        return out

    def bottom_to_top(self) -> Iterable[T]:
        if self.head is not None:
            for x in reversed(self.tail):
                yield x
            yield self.head
