
from pydantic import BaseModel
from pydantic_ai.models import Model

from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.model import SamplingModel
from docassist.sampling.protocols import SamplingStrategy, SamplingController, Sampler, \
    SampleGroupFactory
from docassist.sampling.std.controller import StandardSamplingController
from docassist.sampling.std.provider import SlotProvider


class StandardSampler[K: BaseModel, V: BaseModel](Sampler[K, V]):
    def __init__(self, group_factory: SampleGroupFactory, sampling_strategy: SamplingStrategy | None = None):
        self._controller = StandardSamplingController(sampling_strategy or ConstantSamplingStrategy())
        self._provider = SlotProvider(self._controller, group_factory)

    def controller(self) -> SamplingController:
        return self._controller

    def over_model(self, model: Model, name: str | None = None) -> Model:
        i = 0
        candidates = ["name", "model_name"]
        while name is None:
            try:
                name = getattr(model, candidates[i])
            except AttributeError:
                pass
            i += 1
        assert name is not None #todo better exception
        return SamplingModel(model, self._provider)

