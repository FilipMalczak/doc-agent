from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ContextManager

from pydantic import BaseModel

from docassist.sampling.protocols import SamplingSlot, SamplingController, SamplingStrategy
from docassist.sampling.slots.deferred import DeferredScope
from docassist.sampling.std.util import _stack

SlotProviderAdvice = Callable[[SamplingSlot], SamplingSlot]

class StandardSamplingController(SamplingController):
    def __init__(self, strategy: SamplingStrategy):
        self._advices: _stack[SlotProviderAdvice] = _stack()
        self._strategies: _stack[SamplingStrategy] = _stack()
        self._strategies.push(strategy)
        self.base_dir: Path = None


    @contextmanager
    def defer_until_success(self) -> ContextManager:
        try:
            scope = DeferredScope()
            self._advices.push(scope.wrap)
            yield
            scope.flush()
        except:
            scope.reset()
            raise
        finally:
            self._advices.pop()

    @contextmanager
    def strategy(self, s: SamplingStrategy) -> ContextManager:
        try:
            self._strategies.push(s)
            yield
        finally:
            assert self._strategies.pop() == s
