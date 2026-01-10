import uuid
from typing import overload, NamedTuple

#fixme this module is untested, though it's not rocket science and there's not much to test here anyway

META_NAMESPACE_UUID = uuid.UUID('42afa334-865e-4a0f-b169-ce17a5acfe10')
"""
This is UUID that is used as namespace to generate UUID5 for "Namespace", thus "meta" prefix.

It is an arbitrary UUID4 value, in the spirit of uuid.NAMESPACE_... constants (even though
those constants are UUID1s).  
"""

NAMESPACE = uuid.uuid5(META_NAMESPACE_UUID, "Namespace")

UUID_TO_KNOWN_NAME = {

}

class NamespacedName(NamedTuple):
    namespace: str
    name: str

@overload
def make_id(namespace: str, name: str) -> str: ...

@overload
def make_id(namespace_name: NamespacedName) -> str: ...

def make_id(namespace_or_namespaced_name: str | NamespacedName, name: str=None) -> str:
    if isinstance(namespace_or_namespaced_name, NamespacedName):
        assert name is None
        return make_id(*namespace_or_namespaced_name)
    namespace = namespace_or_namespaced_name
    assert isinstance(namespace, str), f"Namespace: {namespace} / {type(namespace)}"
    assert isinstance(name, str), f"Name: {name} / {type(name)}"
    prefix = uuid.uuid5(NAMESPACE, namespace)
    result = str(uuid.uuid5(prefix, name))
    UUID_TO_KNOWN_NAME[str(prefix)] = NamespacedName("Namespace", namespace)
    UUID_TO_KNOWN_NAME[result] = NamespacedName(namespace, name)
    return result

def resolve_id(uuid: str) -> NamespacedName:
    return UUID_TO_KNOWN_NAME[uuid] #todo better exception than keyerror