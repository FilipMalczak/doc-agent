from pathlib import Path

from docassist.directories import data_dir
from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.protocols import Sampler
from docassist.sampling.std.sampler import StandardSampler
from docassist.sampling.slots.fs import FSGroupFactory

def get_sampler(base_dir: str | Path | None = None) -> Sampler:
    if base_dir is None:
        base_dir = data_dir / "samples"
    return StandardSampler(
        FSGroupFactory(
            Path(base_dir)
        ),
        ConstantSamplingStrategy()
    )