from itertools import product
from typing import Any, Awaitable, Callable, Iterator, Literal

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import Model

from docassist.config import CONFIG
from docassist.sampling.protocols import SamplingController
from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import PromptingTask, writer_system_prompt, doer_system_prompt, \
    solver_system_prompt, Perspective, Example


async def call_agent[I, O: BaseModel](sampling: SamplingController, agent: Agent[Any, O],
                                      input: I, output_type: type[O], **kwargs) -> O:
    assert "user_prompt" not in kwargs, "User prompt is based on input, don't try to set it"
    async with sampling.defer_until_success():
        if isinstance(input, str):
            user_prompt = input
        elif isinstance(input, BaseModel):
            user_prompt = to_simple_xml(
                input.model_dump(mode="json") #fixme this might not be BaseModel!;
                                              # see system_prompts.Example.to_prompt_dict._dump,
                                              # it could become an overlay over pydantic
            )
        else:
            adapter = TypeAdapter(type(input))
            user_prompt = to_simple_xml(
                adapter.dump_python(input, mode="json")
            )
        response = await agent.run(user_prompt=user_prompt, output_type=output_type,  **kwargs)
        return response.output

class StructuredAgent[I, O]:

    def __init__(self, *,
                 name: str, system_prompt: dict[str, Any],
                 input_type: type[I] = Any, output_type: type[O] = None,
                 model: Model | None = None, sampling: SamplingController | None = None):

        self.pydantic_agent = Agent(
            model=model or CONFIG.text_model,
            name=name,
            output_type=output_type,
            retries=2,
            output_retries=3,
            system_prompt=to_simple_xml(system_prompt)
        )
        self.sampling = sampling or CONFIG.sampler.controller()
        self.input_type = input_type
        self.output_type = output_type

    def run[O](self, input: I, output_type: type[O] | None = None, **kwargs) -> Awaitable[O]:
        real_output_type = output_type or self.output_type
        assert real_output_type is not None
        return call_agent(self.sampling, self.pydantic_agent, input, real_output_type, **kwargs)

class WriterAgent[I](StructuredAgent[I, str]):
    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 perspective: Perspective | None, examples: list[Example[I, str]] | None = None,
                 input_type: type[I] = Any,
                 output_format: str | None = None):
        super().__init__(
            name=name,
            system_prompt=writer_system_prompt(
                persona=persona, 
                task=task, 
                perspective=perspective, 
                examples=examples, 
                output_format=output_format
            ),
            input_type=input_type,
            output_type=str,
            model=CONFIG.text_model
        )


class DoerAgent[I, O](StructuredAgent[I, O]):
    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 perspective: Perspective | None, examples: list[Example[I, O]] | None = None,
                 input_type: type[I], output_type: type[O] | None = None):
        super().__init__(
            name=name,
            system_prompt=doer_system_prompt(
                persona=persona, 
                task=task, 
                perspective=perspective, 
                examples=examples
            ),
            input_type=input_type,
            output_type=output_type,
            model=CONFIG.text_model
        )


class SolverAgent[I, O](StructuredAgent[I, O]):
    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 perspective: Perspective | None, examples: list[Example[I, O]] | None = None,
                 input_type: type[I], output_type: type[O] | None = None,
                 ):
        super().__init__(
            name=name,
            system_prompt=solver_system_prompt(
                persona=persona, 
                task=task, 
                perspective=perspective, 
                examples=examples
            ),
            input_type=input_type,
            output_type=output_type,
            model=CONFIG.tool_model
        )

ParameterValue = str | int | bool
NamedParams = dict[str, ParameterValue]
PositionalParams = tuple[ParameterValue, ...]

def expand_on(t) -> list[ParameterValue]:
    # t must be a Literal, but expressing it with typing will suck balls
    return list(t.__args__)

def cross_product(options: dict[str, list[ParameterValue]]) -> Iterator[NamedParams]:
    keys = list(options)
    for values in product(*(options[k] for k in keys)):
        yield dict(zip(keys, values))

class ParametrizedAgent[I, O]:
    def __init__(self, parameters: list[NamedParams], factory: Callable[[NamedParams], StructuredAgent[I, O]]):
        self.parameters = tuple(parameters) #todo make immutable
        self._factory = factory
        self._param_order: tuple[str] = self._figure_out_order()
        self._positional_params: set[PositionalParams] = set([
            self._to_positional(p)
            for p in parameters
        ])
        self._instances: dict[PositionalParams, StructuredAgent[I, O]] = {}

    def _figure_out_order(self) -> tuple[str, ...]:
        all_keys = set()
        for p in self.parameters:
            all_keys.update(p.keys())
        return tuple(sorted(all_keys))

    def _to_positional(self, params: NamedParams) -> PositionalParams:
        return tuple( params[x] for x in self._param_order if x in params )

    def _formatted(self, params: NamedParams) -> str:
        return ", ".join(f"{x}={params[x]}" for x in self._param_order if x in params)

    def parametrized_with(self, params: NamedParams) -> StructuredAgent[I, O]:
        return self.parametrized_as(**params)

    def parametrized_as(self, **params: NamedParams) -> StructuredAgent[I, O]:
        if not self.has_agent(params):
            raise KeyError(f"Parameters {params} not found for this parametrized agent")
        positional = self._to_positional(params)
        if not positional in self._instances:
            instance = self._factory(**params)
            instance.pydantic_agent.name += f" parametrized as {self._formatted(params)}"
            self._instances[positional] = instance
        return self._instances[positional]

    def __truediv__(self, other):
        return self.get_agent(other)

    def has_agent(self, params: NamedParams) -> bool:
        return self._to_positional(params) in self._positional_params

    def __contains__(self, item):
        return self.has_agent(item)