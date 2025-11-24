import hashlib
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, TypeAdapter

from docassist.sampling.protocols import SamplingSlot, SampleGroup, SampleGroupFactory
from docassist.sampling.slots._fsio import FSIO


class FSSamplingSlot[V: BaseModel](SamplingSlot[V]):
    def __init__(self, parent: Path, sample_id: int, fsio: FSIO, sampled_type: type[V]):
        self._sample_id = sample_id
        self._fsio = fsio
        self._storage = parent / f"{sample_id}.{fsio.extension(sampled_type)}"
        self._sampled_type = sampled_type

    def sample_id(self) -> int:
        return self._sample_id

    def sample_coordinates(self) -> str:
        #fixme ugly hack, but it works for now
        from docassist.config import CONFIG
        return str(self._storage.relative_to(CONFIG.samples_dir))

    def get(self) -> V | None:
        if not self._storage.exists():
            return None
        with self._storage.open(self._fsio.read_mode(self._sampled_type)) as f:
            return self._fsio.load(f, self._sampled_type)

    def set(self, val: V | None):
        if val is None:
            self.clear()
            return
        self._storage.parent.mkdir(parents=True, exist_ok=True)
        with self._storage.open(self._fsio.write_mode(self._sampled_type)) as f:
            self._fsio.dump(val, f)

    def clear(self):
        self._storage.unlink(missing_ok=True)

    def is_empty(self) -> bool:
        return not self._storage.exists()


class FSSampleGroup[K: BaseModel, V: BaseModel](SampleGroup[K, V]):
    """
    Samples are stored as:

        * <base_dir>
          * <key_hash>
            * <clash_idx>
              * key.<ext> # keeps dumped key
              * <qualifier> # will be multi-level for names with slash like 'openai/gpt4/generate'
                * <sample_id>.<ext> # keeps dumped value of given sample

    where `ext` depends on the serialization implementation.
    `clash_idx` is an arbitrary value (impl note: consecutive int). When two keys have the same hash, this value is used
      to provide a separate subdirectory for each. That's why we store key in this directory (to compare).
    """
    def __init__(self, base_dir: Path, key: K, value_type: type[V], key_hash: int, qualifier: str, fsio: FSIO):
        self._base_dir = Path(base_dir)
        self._value_type = value_type
        self._key = key
        self._key_hash = key_hash
        self._qualifier = qualifier
        self._fsio = fsio
        self._group_dir = self._resolve_group_dir()
        self._key_file = self._group_dir / f"key.{self._fsio.extension(type(key))}"

        # Write key.txt (overwrite only if this is a new group)
        if not self._key_file.exists():
            with self._key_file.open(self._fsio.write_mode(type(key))) as f:
                self._fsio.dump(key, f)
        self._model_dir = self._group_dir / qualifier
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_group_dir(self) -> Path:
        """Return unique directory path for this key (handle hash clashes)."""
        hash_dir = self._base_dir / str(self._key_hash)
        hash_dir.mkdir(parents=True, exist_ok=True)

        # Check existing clash indices
        existing = sorted(p for p in hash_dir.iterdir() if p.is_dir())
        empties = []
        for clash_dir in existing:
            key_file = clash_dir / f"key.{self._fsio.extension(type(self._key))}"
            if key_file.exists():
                with key_file.open(self._fsio.read_mode(type(self._key))) as f:
                    # if self._fsio.load(f, type(self._key)) == self._key:
                    if self._fsio.dumps(self._key).strip() == f.read().strip():
                        return clash_dir
            else:
                #this may happen because we are happily doing makedirs early; IDGAF, it makes impl cleaner
                empties.append(clash_dir)

        # No match found -> reuse create new clash directory
        if empties:
            return empties[0]
        new_idx = len(existing)
        new_dir = hash_dir / str(new_idx)
        new_dir.mkdir(parents=True)
        return new_dir

    def key(self) -> str:
        return self._key

    def value_type(self) -> type[V]:
        return self._value_type

    def existing_samples(self) -> list[SamplingSlot[V]]:
        slots = []
        for sub in sorted(self._model_dir.iterdir()):
            if sub.is_dir() and sub.name.isdigit():
                slot = FSSamplingSlot(sub, int(sub.name), self._fsio, self._value_type)
                if not slot.is_empty():
                    slots.append(slot)
        return slots

    def new_sample(self, i: int | None = None) -> SamplingSlot[V]:
        existing_ids = [slot.sample_id() for slot in self.existing_samples()]
        if i is not None:
            assert i not in existing_ids #todo better exception
            next_id = i
        else:
            next_id = (max(existing_ids) + 1) if existing_ids else 0
        path = self._model_dir / str(next_id)
        path.mkdir(parents=True, exist_ok=True)
        return FSSamplingSlot(path, next_id, self._fsio, self._value_type)

    def first_gap_sample(self) -> SamplingSlot[V]:
        existing_ids = sorted(slot.sample_id() for slot in self.existing_samples())

        # find first missing integer (0, 1, 2, ...)
        next_id = 0
        for i in existing_ids:
            if i != next_id:
                break
            next_id += 1

        path = self._model_dir / str(next_id)
        path.mkdir(parents=True, exist_ok=True)
        return FSSamplingSlot(path, next_id, self._fsio, self._value_type)

class Hasher(Protocol):
    def hash[T](self, subject: T) -> int: ...

class SHA1OfJsonTrimmed(Hasher):
    def __init__(self, l: int = 32):
        self._l = l

    def hash[T](self, subject: T) -> int:
        dumpable = TypeAdapter(type(subject)).dump_python(subject, mode="json")
        #don't assume T is BaseModel; use typeadapter to cover str, int, etc
        #don't use dump_json - it has no sort_keys param, which we use to try to ensure determinism
        txt = json.dumps(dumpable, sort_keys=True)
        b = txt.encode("utf-8")
        sha1_hash = hashlib.sha1(b)
        return int(sha1_hash.hexdigest()[-self._l:], 16)

class FSGroupFactory(SampleGroupFactory):
    def __init__(self,  base_dir: Path, fsio: FSIO | None = None, hasher: Hasher | None = None):
        self._base_dir = base_dir
        self._fsio = fsio
        self._hasher = hasher or SHA1OfJsonTrimmed()

    def create[K: BaseModel, V: BaseModel](self, key: K, value_type: type[V], qualifier: str) -> SampleGroup[K, V]:
        return FSSampleGroup(self._base_dir, key, value_type, self._hasher.hash(key), qualifier, self._fsio)