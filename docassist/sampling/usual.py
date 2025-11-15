from pathlib import Path

from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.protocols import Sampler
from docassist.sampling.std.sampler import StandardSampler
from docassist.sampling.slots.fs import FSGroupFactory


def get_sampler(base_dir: str | Path = "./response_samples") -> Sampler:
    return StandardSampler(
        FSGroupFactory(
            Path(base_dir)
        ),
        ConstantSamplingStrategy()
    )