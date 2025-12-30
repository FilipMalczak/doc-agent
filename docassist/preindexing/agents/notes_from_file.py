from typing import TypedDict, Self

from pydantic import BaseModel

from docassist.index.document import SourceDocumentType, FileSubjectType, SourceMeta, Document
from docassist.preindexing.perspectives import perspective, AudienceRole, AudienceToProjectRelationship, PERSPECTIVES
from docassist.structured_agent import WriterAgent, ParametrizedAgent, expand_on, cross_product
from docassist.subjects import RepoItemType
from docassist.system_prompts import PromptingTask


class FullyDescribedFile(BaseModel):
    document_type: SourceDocumentType
    type: FileSubjectType
    repo_item_type: RepoItemType
    path: str
    language: str
    content: str

    @classmethod
    def of(cls, doc: Document[SourceMeta]) -> Self:
        data = dict(doc.metadata)
        data["content"] = doc.content
        return cls(**data)

file_note_taker = ParametrizedAgent(
    PERSPECTIVES,
    lambda role, relationship_to_project:
        WriterAgent(
        name="note taker that handles single file",
        persona="note taker",
        perspective=perspective(role, relationship_to_project),
        task=PromptingTask(
            high_level="take notes from the input file",
            low_level="take notes from the perspective of the user of this project",
            detailed="take notes that can be later used to prepare user-facing documentation of the project "
                     "that the input file is part of",
            context="you're reading the whole project for the first time; you need to extract the useful information for later usage"
        ),
        input_type=FullyDescribedFile,
        output_format="Markdown"
    )
)