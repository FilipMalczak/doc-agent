from os import PathLike
from typing import Protocol, Self, Any

from pydantic import BaseModel
import numpy as np

JSONPrimitive = int | float | bool | str | bytes
JSONArray = list["JSON"]
JSONObject = dict[str, "JSON"]
JSON = JSONPrimitive | JSONArray | JSONObject

DocumentId = str
Text = str
Content = Text
Query = Text
Distance = float


class Document(BaseModel):
    id: DocumentId
    content: Content
    metadata: Any #todo this should be JSON



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


class IndexSnapshot(BaseModel):
    path: PathLike
    index: "DocumentIndex"

    class Config:
        arbitrary_types_allowed = True

class DocumentIndex(Protocol):
    @property
    def embedder(self) -> Embedder: ...

    async def add(self, documents: list[Document]): ...

    async def query(self, queries: str, per_query: int) -> list[Document]: ...

    async def get(self, ids: list[DocumentId]) -> list[Document]: ...

    async def store(self, path: PathLike) -> IndexSnapshot: ...

    @classmethod
    async def load(cls, path: PathLike) -> Self: ...
