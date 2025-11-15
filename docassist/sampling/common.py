from pydantic import BaseModel

from docassist.sampling.protocols import SamplingStrategy, SampleGroup, SamplingSlot


class ConstantSamplingStrategy(SamplingStrategy):
    def __init__(self, index: int = 0):
        self._index = index

    def pick_slot[K: BaseModel, V: BaseModel](self, group: SampleGroup[K, V]) -> SamplingSlot[V]:
        return group[self._index] or group.new_sample(self._index)