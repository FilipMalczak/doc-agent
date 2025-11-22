from docassist.index.protocols import Embedder, Embeddings
from docassist.sampling.protocols import SamplingSlot
from docassist.sampling.std.provider import SlotProvider


class SamplingEmbedder(Embedder):
    def __init__(self, delegate: Embedder, slot_provider: SlotProvider):
        self._delegate = delegate
        self._provider = slot_provider

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dimension(self) -> int:
        return self._delegate.dimension

    async def get_embeddings(self, content: str) -> Embeddings:
        slot: SamplingSlot[str, Embeddings] = self._provider.get_slot(content, Embeddings, self._delegate.model_name)
        if slot.is_empty():
            out = await self._delegate.get_embeddings(content)
            slot.set(out)
            return out
        return slot.get()
