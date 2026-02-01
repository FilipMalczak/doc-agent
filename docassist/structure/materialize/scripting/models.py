from typing import Literal

from pydantic import BaseModel, Field

from docassist.index.document import Document

SubjectType = Literal["chapter_preamble", "article_body", "chapter_afterword"]

class ResearchSubject(BaseModel):
    title: str
    subject_type: SubjectType
    description: str

#TODO revew all the models, add descriptions to fields where it makes sense

class ResearchResults(BaseModel):
    plan_as_text: str = Field(
        ...,
        description="""
Plain-text hierarchical plan for the outcome.

STRICT RULES:
- Use section headers, subsections, and bullet points only.
- Use noun phrases or infinitive verb phrases (e.g. "Explain X", "Describe Y").
- DO NOT write full sentences.
- DO NOT include examples, definitions, or narrative prose.
- DO NOT include stylistic or tone-related language.
- This is a semantic structure contract, not a draft.
    """
    )

    document_ids: list[str] = Field(
        ...,
        description="""
List of document identifiers that together contain all information
required to write the outcome correctly and completely. Do not include document type or metadata, just `document_id`s
"""
    )

    notes: str | None = Field(
        None,
        description="""
Optional planning notes, assumptions, or scope decisions.
May use full sentences, but MUST NOT contain outcome prose.
"""
    )

class ManuscriptPlan(BaseModel):
    subject: ResearchSubject
    plan_as_text: str
    notes: str | None
    knowledge: list[Document]