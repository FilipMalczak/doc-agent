from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel

from docassist.agents.generators.facts_from_file import Fact
from docassist.index.document import Document, TransientDocumentType, CommonMetadata, TransientMeta
from docassist.simple_xml import to_simple_xml


class VariableValuation(BaseModel):
    value: str
    explanation: str

class Ancestor(BaseModel):
    title: str
    preamble: str | None

@dataclass
class MaterializationState:
    """
    This is deliberately not pydantic model; its a stateholder, not a serializable thing.
    Pick data from here and put it in prompt inputs or add them to RAG candidates
    """
    ancestors: list[Ancestor]
    variable_values: dict[str, VariableValuation]
    facts: list[Fact]

    def format_text_template(self, txt: str | None) -> str | None:
        if txt is None:
            return None
        return txt.format_map({k: v.value for k, v in self.variable_values.items()})

    def add_ancestor(self, name: str, preamble: str) -> Self:
        return MaterializationState(
            ancestors=self.ancestors + [
                Ancestor(
                    title=name,
                    preamble=preamble
                )
            ],
            variable_values={k: v.model_copy() for k, v in self.variable_values.items()},
            facts=[f.model_copy() for f in self.facts]
        )

    def set_variable(self, name: str, explained_val: VariableValuation) -> Self:
        new_vars = {k: v.model_copy() for k, v in self.variable_values.items()}
        new_vars[name] = explained_val.model_copy()
        return MaterializationState(
            ancestors=list(self.ancestors),
            variable_values=new_vars,
            facts=[f.model_copy() for f in self.facts]
        )

    def add_fact(self, fact: str, explanation: str) -> Self:
        return MaterializationState(
            ancestors=list(self.ancestors),
            variable_values={k: v.model_copy() for k, v in self.variable_values.items()},
            facts=self.facts + [ Fact(fact=fact, explanation=explanation) ]
        )

    def fact_docs(self) -> list[Document]:
        return [
            Document(
                id=f"transient_{i}",
                content=to_simple_xml(f.model_dump(mode="json")),
                metadata=TransientMeta(document_type="transient")
            )
            for i, f in enumerate(self.facts)
        ]

class VariableSpecification(BaseModel):
    name: str
    description: str

class RephraseDescriptionInput(BaseModel):
    desired_rewrite_count: int
    variable_specification: VariableSpecification
    specification_context: list[Ancestor]
    resolved_variables: dict[str, VariableValuation]
    #deliberatly no facts; these should be used for RAG candidates

class RephraseDescriptionOutput(BaseModel):
    rewrites: list[str]

ResolveVariableOutput = VariableValuation
#
# class IndexedAnswer(BaseModel):
#     index: int
#     answer: str
#
# class ExpandAnswersInput(BaseModel):
#     expansions_count: int
#     question: str
#     answers: list[IndexedAnswer]
#
# class ExpandAnswersOutput(BaseModel):
#     expanded: list[str]
#
# class ChooseTheAnswerInput(BaseModel):
#     question: str
#     answers: list[IndexedAnswer]
#     resources: list[DocumentChunk]
#
# class ChooseTheAnswerOutput(BaseModel):
#     chosen_index: int
#     explanation: str
#
# class ExpandOnDomainInput(BaseModel):
#     domain_description: str
#     variables: dict[str, str]
#     resources: list[DocumentChunk]
#
# class IndexedExpansionResult(BaseModel):
#     index: int
#     values: dict[str, VariableValuation]
#
# class ExpandOnDomainOutput(BaseModel):
#     results: list[IndexedExpansionResult]