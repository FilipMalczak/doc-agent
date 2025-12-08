from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from docassist.index.faiss import FAISSIndex
from docassist.index.openai_embedder import OpenAIEmbedder
from docassist.index.protocols import Embedder, DocumentIndex
from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.protocols import SamplingStrategy, Sampler
from docassist.sampling.slots._fsio import FSIO, SpecializedFSIO, YAMLFSIO, PickleFSIO
from docassist.sampling.slots.fs import FSGroupFactory, SHA1OfJsonTrimmed
from docassist.sampling.std.sampler import StandardSampler

load_dotenv()
from os import getenv
from typing import NamedTuple, Self, TypedDict

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from docassist.subjects import AnalysedRepo, PipOutdatedRepo

class EmbedderConfig(TypedDict):
    model_name: str
    dimension: int | None = None


#fixme this is gonna be a bitch to do properly
repo_dir: Path = Path(__file__).parent.parent
data_dir: Path = repo_dir / "data"

class Config(NamedTuple):
    raw_model: Model = OpenAIChatModel(
        'openai/gpt-oss-120b',
        provider=OpenRouterProvider(api_key=getenv("OPENAI_API_KEY")),
    )
    embedder_params: dict[str, str | int | None] = {"model_name": "text-embedding-3-small"}
    samples_dir: Path = Path(data_dir / "samples")

    fsio: FSIO = SpecializedFSIO(
        YAMLFSIO(),
        [
            (lambda t: issubclass(t, np.ndarray), PickleFSIO())
        ]
    )
    hasher = SHA1OfJsonTrimmed() #todo this should also be specialized delegate; we dont wanna hash ndarrays as JSON
    default_sample_id: int = 1
    # default_sample_id: int = 0
    sampling_strategy: SamplingStrategy = ConstantSamplingStrategy(default_sample_id)
    sampler: Sampler = StandardSampler(
        FSGroupFactory(
            Path(samples_dir),
            fsio,
            SHA1OfJsonTrimmed()
        ),
        sampling_strategy
    )

    model: Model = sampler.over_model(raw_model)
    repos: list[AnalysedRepo] = [
        PipOutdatedRepo()
    ]
    async_openai: AsyncOpenAI = raw_model.client
    embedder_config: EmbedderConfig = EmbedderConfig(**embedder_params)
    raw_embedder: Embedder = OpenAIEmbedder(async_openai, **embedder_config)
    embedder: Embedder = sampler.over_embedder(raw_embedder)

    indices_dir: str = Path(data_dir / "indices")

CONFIG = Config()