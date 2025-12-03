from os import PathLike
from typing import Protocol, Self, Any, Annotated, TypedDict

from pydantic import BaseModel, SkipValidation
import numpy as np

from docassist.index.document import Document

JSONPrimitive = int | float | bool | str | bytes
JSONArray = list["JSON"]
JSONObject = dict[str, "JSON"]
JSON = JSONPrimitive | JSONArray | JSONObject

DocumentId = str
Text = str
Content = Text
Query = Text
Distance = float

Embeddings = np.ndarray


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def get_embeddings(self, content: str) -> Embeddings: ...


class SearchResult(BaseModel):
    document: Document
    distance: Distance

#fixme this was supposed to be a smart trick for node caching; not sure if its worth it, but at least it fixes the "move live objects across pydantic graphs" issue
class IndexSnapshot(BaseModel):
    path: PathLike
    index: Annotated["DocumentIndex", SkipValidation()]

    class Config:
        arbitrary_types_allowed = True

class DocumentIndex(Protocol):
    @property
    def embedder(self) -> Embedder: ...

    async def add(self, documents: list[Document]): ...

    async def query(self, queries: str | list[str], total_results: int | None = None) -> list[SearchResult]: ...

    async def get(self, ids: list[DocumentId]) -> list[Document]: ...

    async def store(self, path: PathLike) -> IndexSnapshot: ...

    @classmethod
    async def load(cls, path: PathLike) -> Self: ...
