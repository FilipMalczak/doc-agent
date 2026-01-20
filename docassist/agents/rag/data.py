from typing import Self

from pydantic import BaseModel, Field, AliasChoices

from docassist.index.document import Document
from docassist.index.protocols import SearchResult


class RephrasingInput(BaseModel):
    purpose: str
    rewrite_count: int
    expansion_count: int
    initial_queries: list[str]
    additional_instructions: str


class RephrasingOutput(BaseModel):
    rewrites: list[list[str]]
    expansions: list[str]

    def total(self) -> list[str]:
        def i():
            for r in self.rewrites:
                yield from r
            yield from self.expansions
        return list(i())


class ScoredDocument(BaseModel):
    score: float
    document: Document

    @classmethod
    def from_search_result(cls, x: SearchResult) -> Self:
        return cls(document=x.document, score=1.0-x.distance)

    def rescore(self, new_score: float) -> Self:
        return ScoredDocument(score=new_score, document=self.document)


class IndexedScoredDocument(BaseModel):
    index: int
    score: float
    document: Document

    @classmethod
    def from_scored_documents(cls, docs: list[ScoredDocument]) -> Self:
        return [ cls(index=i, score=x.score, document=x.document) for i, x in enumerate(docs) ]


class DeduplicationInput(BaseModel):
    purpose: str
    documents: list[IndexedScoredDocument]
    additional_instructions: str


class DeduplicationOutput(BaseModel):
    document_id: str
    new_score: float
    explanation: str = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))



class RerankingInput(BaseModel):
    purpose: str
    documents: list[ScoredDocument]
    additional_instructions: str

class RerankedItem(BaseModel):
    item_index: int = Field(validation_alias=AliasChoices("document_index", "idx", "index", "item_idx", "document_idx", "doc_idx", "i"))
    new_score: int = Field(validation_alias=AliasChoices("new_score", "updated_score", "score", "rescored_score", "rescore", "rescored"))
    explanation: str = Field(validation_alias=AliasChoices("explanation", "reason", "justification"))

class RerankingOutput(BaseModel):
    # ordered_document_ids: list[str] #fixme get rid of this
    rescoring: list[RerankedItem]