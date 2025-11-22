
from pydantic import BaseModel
from pydantic_ai.models import Model

from docassist.index.protocols import Embedder
from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.embedder import SamplingEmbedder
from docassist.sampling.model import SamplingModel
from docassist.sampling.protocols import SamplingStrategy, SamplingController, Sampler, \
    SampleGroupFactory
from docassist.sampling.std.controller import StandardSamplingController
from docassist.sampling.std.provider import SlotProvider


class StandardSampler(Sampler):
    def __init__(self, group_factory: SampleGroupFactory, sampling_strategy: SamplingStrategy | None = None):
        self._controller = StandardSamplingController(sampling_strategy or ConstantSamplingStrategy())
        self._provider = SlotProvider(self._controller, group_factory)

    def controller(self) -> SamplingController:
        return self._controller

    def over_model(self, model: Model, name: str | None = None) -> Model:
        return SamplingModel(model, self._provider)

    def over_embedder(self, embedder: Embedder) -> Embedder:
        return SamplingEmbedder(embedder, self._provider)

