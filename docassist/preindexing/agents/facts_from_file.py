from pydantic import BaseModel, Field, AliasChoices

from docassist.agents.generators.notes_from_file import FullyDescribedFile

from docassist.preindexing.perspectives import PERSPECTIVES, perspective
from docassist.structured_agent import WriterAgent, ParametrizedAgent
from docassist.system_prompts import PromptingTask



class Fact(BaseModel):
    fact: str  = Field(validation_alias=AliasChoices("fact", "statement"))
    explanation: str  = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))

class Facts(BaseModel):
    facts: list[Fact]



fact_extractor = ParametrizedAgent(
    PERSPECTIVES,
    lambda role, relationship_to_project:
    WriterAgent(
        name="fact extractor that handles single file",
        persona="fact extractor",
        perspective=perspective(role, relationship_to_project),
        task=PromptingTask(
                high_level="extract atomic facts from the input file",
                low_level="given a source file extract a list of facts that can be inferred from it; they should be short and self-enclosed",
                detailed="extract atomic pieces of knowledge that can be later used to prepare user-facing documentation "
                         "of the project that the input file is part of; explain why you think that each statement is a fact; "
                         "facts should include all the necessary details to be understandable without additional context; "
                         "make sure not to make statements like 'there is one <something>', contextualize instead, resulting "
                         "in 'there is one <something> in <this file>'.",
                context="you're reading the whole project for the first time; you need to extract the useful information for later usage"
            ),
        input_type=FullyDescribedFile,
        output_format="Markdown"
    )
)