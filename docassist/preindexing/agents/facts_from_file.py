from pydantic import BaseModel, Field, AliasChoices

from docassist.index.document import Document
from docassist.parametrized import Parametrized
from docassist.preindexing.perspectives import PERSPECTIVES, perspective
from docassist.structured_agent import WriterAgent, DoerAgent
from docassist.system_prompts import PromptingTask



class Fact(BaseModel):
    fact: str  = Field(validation_alias=AliasChoices("fact", "statement"))
    explanation: str  = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))

class Facts(BaseModel):
    facts: list[Fact]



fact_extractor = Parametrized(
    PERSPECTIVES,
    lambda name_suffix, params:
        DoerAgent(
            name="fact extractor that handles single file"+name_suffix,
            persona="fact extractor",
            perspective=perspective(**params),
            task=PromptingTask(
                    high_level="extract atomic facts from the input file",
                    low_level="given a source file extract a list of facts that can be inferred from it; they should be "
                          "short, self-enclosed and relevant only to the required perspective; you should not look into "
                          "the dependent and depending files, your sole focus should be on currently processed file",
                    detailed="extract atomic pieces of knowledge that can be later used to prepare user-facing documentation "
                         "of the project that the input file is part of; explain why you think that each statement is a fact; "
                         "facts should include all the necessary details to be understandable without additional context; "
                         "make sure not to make statements like 'there is one <something>', contextualize instead, resulting "
                         "in 'there is one <something> in <this file>'; avoid indirect statements like 'in current file' "
                         "in favour of very explicit statements like 'in document with ID <specific ID>'; at this stage "
                         "you should extract facts meaningful to a single perspective (this will happen for any possible "
                         "perspective in parallel); do not note things that are not "
                         "applicable to the current perspective; if there are no facts to be extracted from given perspective,"
                         "use the `empty_result` tool to indicate and explain that. In other case, emit the response "
                         "aligned with the provided schema via `final_result` tool.",
                    context="you're reading the whole project for the first time; you need to extract the useful information for later usage"
                ),
            input_type=Document,
            output_type=Facts | None
        )
)