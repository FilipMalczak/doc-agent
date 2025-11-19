import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Self

import faiss
import numpy as np

from docassist.index.protocols import DocumentIndex, Embedder, DocumentId, Document, IndexSnapshot

#todo consider switching to faiss-gpu

class FAISSIndex(DocumentIndex):
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._documents: dict[DocumentId, Document] = {}
        self._id_to_index: dict[DocumentId, int] = {}
        self._index_to_id: dict[int, DocumentId] = {}

        self._faiss_index = faiss.IndexFlatL2(embedder.dimension())

        self._next_index = 0

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    async def add(self, documents: list[Document]):
        """Add documents in-place and return self (imperative style)."""
        if not documents:
            return self

        # Get embeddings for new documents
        embeddings_list = []
        for doc in documents:
            embedding = await self._embedder.get_embeddings(doc.content)
            embeddings_list.append(embedding)

        embeddings_array = np.vstack(embeddings_list).astype(np.float32)

        # Add documents to metadata
        for doc in documents:
            self._documents[doc.id] = doc
            self._id_to_index[doc.id] = self._next_index
            self._index_to_id[self._next_index] = doc.id
            self._next_index += 1

        # Add embeddings to FAISS index (in-place)
        self._faiss_index.add(embeddings_array)

    async def query(self, query: str, per_query: int) -> list[Document]:
        """Query the index for similar documents."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        query_embedding = await self._embedder.get_embeddings(query)
        query_array = query_embedding.reshape(1, -1).astype(np.float32)

        # Search FAISS index
        distances, indices = self._faiss_index.search(query_array, min(per_query, self._faiss_index.ntotal))

        # Convert results to documents
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                doc_id = self._index_to_id[idx]
                document = self._documents[doc_id]
                results.append(document)

        return results

    async def get(self, ids: list[DocumentId]) -> list[Document]:
        """Get documents by their IDs."""
        return [self._documents[doc_id] for doc_id in ids if doc_id in self._documents]

    async def store(self, path: PathLike) -> IndexSnapshot:
        """Store the index to filesystem."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._faiss_index is not None:
            faiss_path = path / "index.faiss"
            faiss.write_index(self._faiss_index, str(faiss_path))

        # Save metadata
        metadata = {
            "documents": {doc_id: doc.model_dump() for doc_id, doc in self._documents.items()},
            "id_to_index": self._id_to_index,
            "index_to_id": {str(k): v for k, v in self._index_to_id.items()},  # JSON keys must be strings
            "next_index": self._next_index,
            "dimension": self._faiss_index.d if self._faiss_index is not None else None
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embedder (pickle for simplicity)
        embedder_path = path / "embedder.pkl"
        with open(embedder_path, 'wb') as f:
            pickle.dump(self._embedder, f)

        return IndexSnapshot(path=path, index=self)

    @classmethod
    async def load(cls, path: PathLike) -> Self:
        """Load index from filesystem."""
        path = Path(path)

        # Load embedder
        embedder_path = path / "embedder.pkl"
        with open(embedder_path, 'rb') as f:
            embedder = pickle.load(f)

        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(embedder)

        # Restore documents
        instance._documents = {
            doc_id: Document(**doc_data)
            for doc_id, doc_data in metadata["documents"].items()
        }

        # Restore mappings
        instance._id_to_index = metadata["id_to_index"]
        instance._index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
        instance._next_index = metadata["next_index"]

        # Load FAISS index
        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            instance._faiss_index = faiss.read_index(str(faiss_path))
        elif metadata["dimension"] is not None:
            instance._faiss_index = faiss.IndexFlatL2(metadata["dimension"])

        return instance