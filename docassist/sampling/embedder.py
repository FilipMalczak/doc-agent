import logfire

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
        #fixme this is pretty half-assed
        with logfire.span(f"embeddings {self.model_name}") as s:
            s.set_attribute("sampling.enabled", True)
            slot: SamplingSlot[str, Embeddings] = self._provider.get_slot(content, Embeddings, self._delegate.model_name)
            s.set_attribute("sampling.slot.sample_id", slot.sample_id())
            s.set_attribute("sampling.slot.sample_coordinates", slot.sample_coordinates())
            s.set_attribute("sampling.slot.empty", slot.is_empty())

            if slot.is_empty():
                s.set_attribute("sampling.action", "delegate")
                out = await self._delegate.get_embeddings(content)
                slot.set(out)
                return out
            s.set_attribute("sampling.action", "reuse")
            return slot.get()
