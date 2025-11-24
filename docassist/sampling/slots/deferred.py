from pydantic import BaseModel

from docassist.sampling.protocols import SamplingSlot

class DeferredSlot[V: BaseModel](SamplingSlot[V]):
    """
    Decorator that defers persistence and deletion until flush().
    """

    def __init__(self, delegate: SamplingSlot):
        self._delegate = delegate
        self._buffered_value: V | None = None
        self._has_buffer = False
        self._pending_clear = False

    def sample_id(self) -> int:
        return self._delegate.sample_id()

    def sample_coordinates(self) -> str:
        return self._delegate.sample_coordinates()

    def get(self):
        if self._pending_clear:
            return None
        if self._has_buffer:
            return self._buffered_value
        delegated = self._delegate.get()
        return delegated

    def set(self, val):
        if val is None:
            # defer deletion
            self._buffered_value = None
            self._has_buffer = False
            self._pending_clear = True
        else:
            self._buffered_value = val
            self._has_buffer = True
            self._pending_clear = False

    def clear(self):
        # same as set(None), for convenience
        self.set(None)

    def is_empty(self) -> bool:
        if self._pending_clear:
            return True
        if self._has_buffer:
            return False
        return self._delegate.is_empty()

    def flush(self):
        """Persist buffered changes or deletions."""
        assert not (self._pending_clear and self._has_buffer)
        if self._pending_clear:
            self._delegate.clear()
        elif self._has_buffer:
            self._delegate.set(self._buffered_value)

        # reset buffer state
        self._buffered_value = None
        self._has_buffer = False
        self._pending_clear = False

    def reset(self):
        """Discard any buffered value or pending deletion without flushing."""
        self._buffered_value = None
        self._has_buffer = False
        self._pending_clear = False


class DeferredScope:
    def __init__(self):
        self.wrappers: set[DeferredSlot] = set([])

    def wrap(self, slot: SamplingSlot) -> SamplingSlot:
        out = DeferredSlot(slot)
        self.wrappers.add(out)
        return out

    def flush(self):
        for x in self.wrappers:
            x.flush()

    def reset(self):
        for x in self.wrappers:
            x.reset()