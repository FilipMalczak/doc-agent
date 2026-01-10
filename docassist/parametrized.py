from itertools import product
from typing import Iterator, Callable

ParameterValue = str | int | bool
NamedParams = dict[str, ParameterValue]
PositionalParams = tuple[ParameterValue, ...]

def expand_on(t) -> list[ParameterValue]:
    # t must be a Literal, but expressing it with typing will suck balls
    return list(t.__args__)

def cross_product(options: dict[str, list[ParameterValue]]) -> Iterator[NamedParams]:
    keys = list(options)
    for values in product(*(options[k] for k in keys)):
        yield dict(zip(keys, values))

NameSuffix = str

class Parametrized[T]:
    def __init__(self, parameters: list[NamedParams], factory: Callable[[NameSuffix, NamedParams], T]):
        self.parameters = tuple(parameters) #todo make immutable
        self._factory = factory
        self._param_order: tuple[str] = self._figure_out_order()
        self._positional_params: set[PositionalParams] = set([
            self._to_positional(p)
            for p in parameters
        ])
        self._instances: dict[PositionalParams, T] = {}

    def _figure_out_order(self) -> tuple[str, ...]:
        all_keys = set()
        for p in self.parameters:
            all_keys.update(p.keys())
        return tuple(sorted(all_keys))

    def _to_positional(self, params: NamedParams) -> PositionalParams:
        return tuple( params[x] for x in self._param_order if x in params )

    def _formatted(self, params: NamedParams) -> str:
        return ",".join(f"{x}={params[x]}" for x in self._param_order if x in params)

    def parametrized_with(self, params: NamedParams) -> T:
        return self.parametrized_by(**params)

    def parametrized_by(self, **params: NamedParams) -> T:
        if not self.has_agent(params):
            raise KeyError(f"Parameters {params} not found for this parametrized instance ({self})")
        positional = self._to_positional(params)
        if not positional in self._instances:
            name_suffix = f"[{self._formatted(params)}]"
            instance = self._factory(name_suffix, params)
            self._instances[positional] = instance
        return self._instances[positional]

    def __truediv__(self, other):
        return self.get_agent(other)

    def has_agent(self, params: NamedParams) -> bool:
        return self._to_positional(params) in self._positional_params

    def __contains__(self, item):
        return self.has_agent(item)