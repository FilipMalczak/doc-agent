from functools import wraps
from typing import Callable, Iterable

from pydantic_ai import FunctionToolset


def is_tool_method[C: Callable](foo: C) -> bool:
    return getattr(foo, "__is_tool_method__", False)

def tool_method[C: Callable](foo: C) -> C:
    foo.__is_tool_method__ = True
    return foo

class Tools:
    def tool_methods(self) -> Iterable[Callable]:
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Callable) and is_tool_method(attr):
                yield attr

    def as_toolset(self) -> FunctionToolset:
        tool_wrappers = []
        for tool in self.tool_methods():
            #todo enforces that all tools are async; allow for standard ones too
            @wraps(tool)
            async def foo(*args, **kwargs):
                return await tool(*args, **kwargs)
            foo.__annotations__ = tool.__annotations__
            foo.__doc__ = tool.__doc__
            tool_wrappers.append(foo)
        return FunctionToolset(tool_wrappers, max_retries=3, require_parameter_descriptions=True)
