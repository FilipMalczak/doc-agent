from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from email.policy import default
from pathlib import Path
from typing import Callable, ContextManager, AsyncContextManager

from pydantic import BaseModel

from docassist.sampling.protocols import SamplingSlot, SamplingController, SamplingStrategy
from docassist.sampling.slots.deferred import DeferredScope

SlotProviderAdvice = Callable[[SamplingSlot], SamplingSlot]

class StandardSamplingController(SamplingController):
    def __init__(self, strategy: SamplingStrategy):
        self._advices: ContextVar[list[SlotProviderAdvice]] = ContextVar("advices", default=[])
        self._strategy: ContextVar[SamplingStrategy] = ContextVar("strategy", default=strategy)
        self.base_dir: Path = None


    @asynccontextmanager
    async def defer_until_success(self) -> AsyncContextManager:
        try:
            scope = DeferredScope()
            new_advices = self._advices.get() + [ scope.wrap ]
            token = self._advices.set(new_advices)
            yield
            scope.flush()
        except:
            scope.reset()
            raise
        finally:
            self._advices.reset(token)

    @asynccontextmanager
    async def strategy(self, s: SamplingStrategy) -> AsyncContextManager:
        try:
            token = self._strategy.set(s)
            yield
        finally:
            self._strategy.reset(token)
