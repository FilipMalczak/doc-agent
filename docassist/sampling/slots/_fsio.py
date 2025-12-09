import pickle
from io import BytesIO, StringIO
from typing import Protocol, IO, Callable

import yaml
from pydantic import BaseModel, TypeAdapter
from yaml import Loader


class FSIO(Protocol):
    def dump[T](self, x: T, f: IO): ...

    def dumps[T](self, x: T) -> str | bytes:
        out = BytesIO() if self.is_binary(type(x)) else StringIO()
        self.dump(x, out)
        return out.getvalue()

    def load[T](self, f: IO, t: type[T]) -> T: ...

    def extension(self, t: type) -> str: ...

    def is_binary(self, t: type) -> bool: ...

    def read_mode(self, t: type) -> str:
        return "rb" if self.is_binary(t) else "r"

    def write_mode(self, t: type) -> str:
        return "wb" if self.is_binary(t) else "w"


class YAMLFSIO(FSIO):
    def dump[T: BaseModel](self, x: T, f: IO):
        yaml.dump(TypeAdapter(type(x)).dump_python(x, mode="json"), f, sort_keys=True)

    def load[T: BaseModel](self, f: IO, t: type[T]) -> T:
        return TypeAdapter(t).validate_python(yaml.load(f, Loader=Loader))

    def extension(self, t: type) -> str: return "yaml"

    def is_binary(self, t: type) -> bool:
        return False

class PickleFSIO(FSIO):
    def dump[T: BaseModel](self, x: T, f: IO):
        pickle.dump(x, f)

    def load[T: BaseModel](self, f: IO, t: type[T]) -> T:
        return pickle.load(f)

    def extension(self, t: type) -> str: return "bin"

    def is_binary(self, t: type) -> bool:
        return True

TypeMatcher = Callable[[type], bool]
Specialization = tuple[TypeMatcher, FSIO]

class SpecializedFSIO(FSIO):
    def __init__(self, default_delegate: FSIO, specializations: list[Specialization]):
        self._delegates = specializations + [(lambda x: True, default_delegate)]

    def add_delegate(self, specialization: Specialization):
        self._delegates.insert(0, specialization)

    def _pick_delegate[T](self, subject: T | type[T]) -> FSIO:
        if not isinstance(subject, type):
            subject = type(subject)
        for m, fsio in self._delegates:
            if m(subject):
                return fsio
        assert False

    def dump[T](self, x: T, f: IO):
        return self._pick_delegate(x).dump(x, f)

    def load[T](self, f: IO, t: type[T]) -> T:
        return self._pick_delegate(t).load(f, t)

    def extension(self, t: type) -> str:
        return self._pick_delegate(t).extension(t)

    def is_binary(self, t: type) -> bool:
        return self._pick_delegate(t).is_binary(t)