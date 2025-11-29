import json
from typing import Any

from pydantic import TypeAdapter

from docassist.simple_xml import to_simple_xml

def data_from_user(d: dict[str, Any]) -> str:
    return to_simple_xml(d)

def object_from_user[T](o: T) -> str:
    adapter = TypeAdapter(type(o))
    data = adapter.dump_python(o, mode="json")
    serialized = data_from_user(data)
    return serialized

def data_from_llm(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2)

def object_from_llm[T](o: T) -> str:
    adapter = TypeAdapter(type(o))
    data = adapter.dump_python(o, mode="json")
    serialized = data_from_llm(data)
    return serialized