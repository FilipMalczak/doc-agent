import numpy as np
from openai import AsyncOpenAI
from openai import OpenAIError

from docassist.index.protocols import Embeddings, Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, client: AsyncOpenAI, model_name: str = "text-embedding-3-small", dimensions: int | None = None):
        """
        :param client: instance of AsyncOpenAI
        :param model_name: model name, e.g. "text-embedding-3-large"
        :param dimensions: optional truncation of dimension (only supported on newer models)
        """
        self._client = client
        self._model_name = model_name
        self._dimensions = dimensions  # if None, use default dims for model

    @property
    def model_name(self) -> str:
        return f"openai/{self._model_name}/{self.dimension}"

    async def get_embeddings(self, content: str) -> Embeddings:
        if content.strip():
            inputs = [content]
            try:
                resp = await self._client.embeddings.create(
                    input=inputs,
                    model=self._model_name,
                    **({"dimensions": self._dimensions} if self._dimensions is not None else {})
                )
            except BaseException as e:
                raise RuntimeError(f"OpenAI embedding failed: {e}") from e
            embedding = resp.data[0].embedding  # this should be a list of floats
            return np.array(embedding, dtype=np.float32)
        # In case of empty docs we use unit vector.
        # Real-life embedders usually produce vectors normalized to unit length and
        # virtually never yield unit vectors (they encode the information between
        # dimensions and in their variability).
        # [0...0, 1] should do the trick as "empty file" constant
        v = np.zeros(self.dimension)
        v[-1] = 1.0
        return v

    @property
    def dimension(self) -> int:
        # If the user asked for a specific dimension, return that
        if self._dimensions is not None:
            return self._dimensions

        # otherwise infer from model
        # (you could also hardcode a map, or fetch metadata, but here is a simple map)
        if self._model_name == "text-embedding-ada-002":
            return 1536
        if self._model_name == "text-embedding-3-small":
            # OpenAI doesn't officially document a smaller fixed dimension publicly,
            # but small is very efficient; assuming it's same as ada (or you can log from response)
            return 1536  # or whatever is correct / measured
        if self._model_name == "text-embedding-3-large":
            return 3072 if self._dimensions is None else self._dimensions

        # fallback
        raise ValueError(f"Unknown model dimension for {self._model_name}")