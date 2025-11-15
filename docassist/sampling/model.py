import datetime
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Iterable, Optional, Any, Awaitable, Literal, AsyncIterator, Callable, Self

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import ModelMessage, ModelSettings, ModelResponse, RequestUsage, RunContext, ModelRequest, \
    ModelRequestPart, ModelResponsePart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse


from docassist.sampling.protocols import SamplingSlot
from docassist.sampling.std.provider import SlotProvider


# Consider:
#
# async def foo(i: int) -> str:
#     ...
#
# async def bar() -> str:
#     return await foo(1)
#
# It is quicker to do
#
# def bar() -> Awaitable[str]:
#     return foo(1)
#
# and save yourself some stack frames.
# This is useful for methods that need quick delegation, but for sampling we need to await afoo variants to persist the
# result.


class ModelInput(BaseModel):
    messages: list[ModelMessage]
    # model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters

    def remove_transient_data(self) -> Self:
        def part[T: ModelRequestPart | ModelResponsePart](p: T) -> T:
            p = replace(p)
            p.timestamp = datetime.datetime.fromtimestamp(0, datetime.UTC)
            # fixme
            return p
        def msg(m: ModelMessage) -> ModelMessage:
            m = replace(m)
            m.parts = [part(p) for p in m.parts]
            #arbitrary constant UUID
            m.run_id = uuid.UUID("9063c696-c25f-11f0-8000-f020ff65b25b")
            return m
        return ModelInput(
            messages=[msg(m) for m in self.messages],
            model_request_parameters=replace(self.model_request_parameters)
        )


class SamplingModel(Model):
    def __init__(self, delegate: Model, slot_provider: SlotProvider):
        self._delegate = delegate
        self._provider = slot_provider

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        inp = ModelInput(
            messages=messages,
            # model_settings=model_settings,
            model_request_parameters=model_request_parameters
        ).remove_transient_data()
        slot: SamplingSlot[ModelInput, ModelResponse] = self._provider.get_slot(inp, ModelResponse, self._delegate.model_name)
        if slot.is_empty():
            out = await self._delegate.request(messages, model_settings, model_request_parameters)
            slot.set(out)
            return out
        return slot.get()


    def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Awaitable[RequestUsage]:
        """Make a request to the model for counting tokens."""
        inp = ModelInput(
            messages=messages,
            # model_settings=model_settings,
            model_request_parameters=model_request_parameters
        ).remove_transient_data()
        slot: SamplingSlot[ModelInput, ModelResponse] = self._provider.get_slot(inp, ModelResponse, self._delegate.model_name)
        if slot.is_empty():
            return self._delegate.count_tokens(messages, model_settings, model_request_parameters)
        async def foo():
            return RequestUsage(details={"sampled"})
        return foo()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        with self._delegate.request_stream(messages, model_settings, model_request_parameters, run_context):
            yield


    @property
    def model_name(self) -> str:
        """The model name."""
        self._delegate.model_name


    @property
    def system(self) -> str:
        """The model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        #todo add sampling details like sampled-in-fs-..., sampled-in-redis-..., etc; that would need the slot impl to expose name (fs, redis, etc)
        return "sampled-"+self._delegate.system
