import types
from itertools import product
from types import NoneType
from typing import Any, Awaitable, Callable, Iterator, Literal, Sequence, Union, get_origin, get_args

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, AbstractToolset, FunctionToolset
from pydantic_ai.models import Model

from docassist.config import CONFIG
from docassist.sampling.protocols import SamplingController
from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import PromptingTask, writer_system_prompt, doer_system_prompt, \
    solver_system_prompt, Perspective, Example
from docassist.tool_exceptions import AgentDoesntKnow, EmptyResult


def unwrap_none[T](t: type[T] | type[T | None]) -> tuple[bool, T]:
    origin = get_origin(t)

    # --- PEP 604 unions: X | Y | None ---
    if isinstance(t, types.UnionType):
        args = get_args(t)
        if NoneType in args:
            remaining = tuple(a for a in args if a is not NoneType)
            if len(remaining) == 1:
                return True, remaining[0]
            return True, remaining[0] | remaining[1] if len(remaining) == 2 else eval(" | ".join(map(str, remaining)))
        return False, t

    # --- typing.Optional / typing.Union ---
    if origin is types.UnionType or origin is None:
        pass

    if origin is not None:
        args = get_args(t)
        if origin is getattr(__import__("typing"), "Union", None):
            # typing.Optional[T] == typing.Union[T, None]
            if NoneType in args and len(args) == 2:
                other = args[0] if args[1] is NoneType else args[1]
                return False, other
        return False, t

    return False, t

async def call_agent[I, O: BaseModel, D](
        sampling: SamplingController, agent: Agent[D, O],
        input: I, output_type: type[O],
        allow_dont_know: bool = False,
        toolsets: Sequence[AbstractToolset[D]] | None = None,
        **kwargs) -> O:
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

        def empty_result(comment: str):
            """
            Indicate that there is no meaningful result to the task you're handling. It means a successful, but empty
            response.

            :param comment: explanation as to why current task has no result
            :return: end of the agent session
            """
            raise EmptyResult(comment)

        def i_dont_know(comment: str):
            """
            Indicate that you cannot provide a correct answer. Use this tool only if you don't know (and have no way to
            get to know) the answer, not when there should be no answer (in which case use the `empty_result` instead).
            Consider this your way to admit defeat instead of hallucinating an answer; using this tool will be considered
            a failure to solve your task.
            
            :param comment: explanation as to why you're unable to provide the result
            :return: end of the agent session
            """

            raise AgentDoesntKnow(comment)
        can_be_empty, non_empty_type = unwrap_none(output_type)
        control_tools = []
        if can_be_empty:
            control_tools.append(empty_result)
        if allow_dont_know:
            control_tools.append(i_dont_know)
        effective_toolsets = [
            FunctionToolset(control_tools, max_retries=1, require_parameter_descriptions=True)
        ]
        if toolsets:
            effective_toolsets.extend(toolsets)
        try:
            response = await agent.run(user_prompt=user_prompt, output_type=output_type, toolsets=effective_toolsets, **kwargs)
            return response.output
        except EmptyResult as e:
            #todo add span attribute to indicate whether it was empty or not; ditto I dont know
            return None

class StructuredAgent[I, O]:

    def __init__(self, *,
                 name: str, system_prompt: dict[str, Any],
                 input_type: type[I] = Any, output_type: type[O] = None,
                 allow_dont_know: bool = False,
                 model: Model | None = None, sampling: SamplingController | None = None):

        self.pydantic_agent = Agent(
            model=model or CONFIG.text_model,
            name=name,
            output_type=output_type,
            retries=5,
            # output_retries=5,
            system_prompt=to_simple_xml(system_prompt)
        )
        self.sampling = sampling or CONFIG.sampler.controller()
        self.input_type = input_type
        self.output_type = output_type
        self.allow_dont_know = allow_dont_know

    def run[O](self, input: I, output_type: type[O] | None = None, **kwargs) -> Awaitable[O]:
        real_output_type = output_type or self.output_type
        assert real_output_type is not None
        return call_agent(self.sampling, self.pydantic_agent, input, real_output_type, self.allow_dont_know, **kwargs)

class WriterAgent[I](StructuredAgent[I, str | None]): #todo str | None depends on allow_no_result but is hardcoded here
    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 perspective: Perspective | None, examples: list[Example[I, str]] | None = None,
                 input_type: type[I] = Any,
                 allow_no_result: bool = False,
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
            output_type=(str | None) if allow_no_result else str,
            model=CONFIG.text_model
        )


class DoerAgent[I, O](StructuredAgent[I, O]):
    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 perspective: Perspective | None, examples: list[Example[I, O]] | None = None,
                 allow_dont_know: bool = False,
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
            allow_dont_know=allow_dont_know,
            model=CONFIG.mixed_model
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
            allow_dont_know=True,
            model=CONFIG.tool_model
        )
