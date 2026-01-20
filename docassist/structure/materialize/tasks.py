from typing import NamedTuple, Any
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter, Field, AliasChoices

from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import PromptingTask

#fixme get rid of named tuples, make everything BaseModels!!!
class MaterializationAideInput[I](BaseModel):
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
            high_level="Resolve value of a variable given its name and description",
            low_level="Given variable name and its description generate the value of the variable. Use available tools "
                      "to retrieve all the required informations from the knowledge base.",
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

class IdentifiableAnswer(BaseModel):
    id: str
    answer: str

class ChooseAnAnswerInput(BaseModel):
    question: str
    answers: list[IdentifiableAnswer]

class ChooseAnAnswerOutput(BaseModel):
    correct_answer_id: str = Field(validation_alias=AliasChoices("answer_id", "correct_answer", "id"))
    explanation: str = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))

#todo add examples section to prompting task

def choose_an_answer(question: str, answers: list[str | IdentifiableAnswer]) -> tuple[MaterializationAideInput, list[str]]:
    """
    :return: tuple(input you feed to the materialization aide, list of answers ids, so you can figure out what the output means)
    """
    #fixme switch to index-based picking
    answers_with_ids = [
        a if isinstance(a, IdentifiableAnswer) else IdentifiableAnswer(id=str(uuid4()), answer=a)
        for a in answers
    ]
    assert len(answers_with_ids) == len(set(x.id for x in answers_with_ids))
    return MaterializationAideInput(
        current_task=PromptingTask(
            context=None,
            high_level="Answer a closed question",
            low_level="Given the question, and the set of answers (each given a specific ID) pick a correct answer. Use "
                      "available tools to retrieve all the required information from the knowledge base.",
            detailed="""
You will be given the question to answer and a list of answers, where each will have unique ID assigned. 
You will infer the result only from the available data. You will not fabricate or come up with any 
facts. You will make sure that the result value can be traced to the knowledge base documents that
you've used. You will respond by providing correct answer ID.
    """
        ),
        input=ChooseAnAnswerInput(
            question=question,
            answers=answers_with_ids
        )
    ), [a.id for a in answers_with_ids]
