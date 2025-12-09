from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import Model

from docassist.config import CONFIG
from docassist.sampling.protocols import SamplingController
from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import PromptingTask, simple_xml_system_prompt


async def call_agent[I, O: BaseModel](sampling: SamplingController, agent: Agent[Any, O],
                                      input: I, output_type: type[O], **kwargs) -> O:
    assert "user_prompt" not in kwargs, "User prompt is based on input, don't try to set it"
    async with sampling.defer_until_success():
        if isinstance(input, str):
            user_prompt = input
        elif isinstance(input, BaseModel):
            user_prompt = to_simple_xml(
                input.model_dump(mode="json")
            )
        else:
            adapter = TypeAdapter(type(input))
            user_prompt = to_simple_xml(
                adapter.dump_python(input, mode="json")
            )
        response = await agent.run(user_prompt=user_prompt, output_type=output_type,  **kwargs)
        return response.output

# generators and materializer aide are still using "raw" agents
class StructuredAgent[I, O]:

    def __init__(self, *,
                 name: str, persona: str | None = None, task: PromptingTask,
                 input_type: type[I] = Any, output_type: type[O] = None,
                 input_format: str | None = None, output_format: str | None = None,
                 model: Model | None = None, sampling: SamplingController | None = None):
        persona = persona or name
        self.pydantic_agent = Agent(
            model=model or CONFIG.model,
            name=name,
            output_type=output_type,
            retries=2,
            output_retries=3,
            system_prompt=simple_xml_system_prompt(
                persona=persona,
                task=task,
                input_format=input_format,
                output_format=output_format
            )
        )
        self.sampling = sampling or CONFIG.sampler.controller()
        self.input_type = input_type
        self.output_type = output_type

    async def run[O](self, input: I, output_type: type[O] | None = None, **kwargs) -> O:
        real_output_type = output_type or self.output_type
        assert real_output_type is not None
        return await call_agent(self.sampling, self.pydantic_agent, input, real_output_type, **kwargs)


