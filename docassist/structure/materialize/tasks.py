from typing import NamedTuple, Any

from pydantic import BaseModel, TypeAdapter, Field, AliasChoices

from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import PromptingTask


class MaterializationAideInput[I](NamedTuple):
    current_task: PromptingTask
    input: I

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "current_task": self.current_task.to_prompt_dict(),
            "input": TypeAdapter(type(self.input)).dump_python(self.input, mode="json")
        }

    def message_content(self) -> str:
        return to_simple_xml(self.to_prompt_dict())

class EvaluateVariableInput(BaseModel):
    problem: str
    variable_name: str
    variable_definition: str

class EvaluateVariableOutput(BaseModel):
    variable_name: str = Field(validation_alias=AliasChoices("name", "variable", "var", "var_name"))
    variable_value: str = Field(validation_alias=AliasChoices("value", "val", "result"))
    explanation: str  = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))

def evaluate_variable(name: str, description: str) -> MaterializationAideInput:
    return MaterializationAideInput(
        current_task=PromptingTask(
            context=None,
            high_level="Resolve variable value and explain your valuation",
            low_level="Given variable name, description and a set of auxiliary documents retrieved from knowledge base "
                      "generate the value of the variable and explain your reasoning.",
            detailed="You will infer the result only from the available data. You will not fabricate or come up with any "
                     "facts. You will make sure that the result value can be traced to the knowledge base documents that"
                     "you've used. You will phrase the result value as simple as possible. If the result is a number, "
                     "you will do your best to write it with digits and not words. If the result is a boolean, you will "
                     "phrase it as python `True`/`False` values. If the result is text, you will ensure it is concise "
                     "and focused."
        ),
        input=EvaluateVariableInput(
            problem=f"Resolve value of variable `{name}` as defined as '{description}'",
            variable_name=name,
            variable_definition=description
        )
    )