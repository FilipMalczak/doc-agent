from typing import Protocol, ContextManager, Iterable, Optional, Any

from pydantic import BaseModel
from pydantic_ai.models import Model


class SamplingSlot[V: BaseModel](Protocol):
    def sample_id(self) -> int: ...

    def get(self) -> V | None: ...

    def set(self, val: V | None): ...

    def clear(self): ...

    def is_empty(self) -> bool: ...

class SampleGroup[K: BaseModel, V: BaseModel](Protocol):
    def key(self) -> K: ...

    def key_type(self) -> type[K]:
        return type(self.key())

    def value_type(self) -> type[V]: ...

    def existing_samples(self) -> list[SamplingSlot[V]]: ...

    def new_sample(self, i: int | None = None) -> SamplingSlot[V]: ...
    def first_gap_sample(self) -> SamplingSlot[V]: ...

    def get_sample(self, i: int) -> SamplingSlot[V] | None:
        for x in self.existing_samples():
            if x.sample_id() == i:
                return x

    def __getitem__(self, item: int) -> SamplingSlot[V]:
        return self.get_sample(item)

class SamplingStrategy(Protocol):
    def pick_slot[K: BaseModel, V: BaseModel](self, group: SampleGroup[K, V]) -> SamplingSlot[V]: ...


class SampleGroupFactory(Protocol):
    def create[K: BaseModel, V: BaseModel](self, key: K, value_type: type[V], qualifier: str) -> SampleGroup[K, V]: ...

class SamplingController(Protocol):
    def defer_until_success(self) -> ContextManager: ...

    def strategy(self, s: SamplingStrategy) -> ContextManager: ...

# class LangChainProtocol(Protocol):
#
#     def invoke(
#         self,
#         input: Iterable[BaseMessage] | list[BaseMessage] | BaseMessage,
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> BaseMessage:
#         """
#         Required by LC Runnable interface.
#         Returns the *assistant* message only (not the full generation).
#         """
#         ...
#
#     async def ainvoke(
#         self,
#         input: Iterable[BaseMessage] | list[BaseMessage] | BaseMessage,
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> BaseMessage:
#         ...
#
#     # ----- OPTIONAL (but used by higher-level chains/tools) -----
#
#     def generate(
#         self,
#         messages: list[list[BaseMessage]],
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """
#         Batched generation. Required by LC constructs that call ChatModel.generate().
#         """
#         ...
#
#     async def agenerate(
#         self,
#         messages: list[list[BaseMessage]],
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         ...
#
#     # ----- INTERNAL HOOKS MODELS ARE EXPECTED TO PROVIDE -----
#
#     @property
#     def _llm_type(self) -> str:
#         """
#         Used for repr/debugging.
#         """
#         ...
#
#     def _combine_llm_outputs(
#         self,
#         llm_outputs: list[Optional[dict[str, Any]]]
#     ) -> Optional[dict[str, Any]]:
#         """
#         Combine per-call provider metadata into a single structure.
#         Many custom LLMs simply return None.
#         """
#         ...
#
#     # ----- (RARELY NEEDED) STREAMING -----
#
#     def stream(
#         self,
#         input: Iterable[BaseMessage] | list[BaseMessage] | BaseMessage,
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> Iterable[BaseMessage]:
#         """
#         Optional, needed only if you want support for .stream().
#         """
#         ...
#
#     async def astream(
#         self,
#         input: Iterable[BaseMessage] | list[BaseMessage] | BaseMessage,
#         *,
#         stop: Optional[list[str]] = None,
#         **kwargs: Any,
#     ) -> Iterable[BaseMessage]:
#         ...

class Sampler[K: BaseModel, V: BaseModel](Protocol):
    def over_model(self, model: Model) -> Model: ...

    def controller(self) -> SamplingController: ...