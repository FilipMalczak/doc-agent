from pydantic import BaseModel

from docassist.sampling.protocols import SampleGroupFactory, SamplingSlot
from docassist.sampling.std.controller import StandardSamplingController


class SlotProvider:
    """
    This class befriends specific implementation of controller. No "standard" prefix, because its implementation-specific.
    """
    def __init__(self, controller: StandardSamplingController, group_factory: SampleGroupFactory):
        self._controller = controller
        self._group_factory = group_factory

    def get_slot[K: BaseModel, V: BaseModel](self, key: K, value_type: type[V], qualifier: str) -> SamplingSlot[V]:
        group = self._group_factory.create(key, value_type, qualifier)
        slot: SamplingSlot[V] = self._controller._strategy.get().pick_slot(group)
        postprocessed_slot = slot
        for advice in self._controller._advices.get():
            postprocessed_slot = advice(postprocessed_slot)
        return postprocessed_slot
