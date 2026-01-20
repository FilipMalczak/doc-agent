import os
from os import getenv
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from openai.types import ReasoningEffort
from pydantic_ai import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from docassist.index.openai_embedder import OpenAIEmbedder
from docassist.index.protocols import Embedder
from docassist.sampling.common import ConstantSamplingStrategy
from docassist.sampling.protocols import SamplingStrategy, Sampler
from docassist.sampling.slots._fsio import FSIO, SpecializedFSIO, YAMLFSIO, PickleFSIO
from docassist.sampling.slots.fs import FSGroupFactory, SHA1OfJsonTrimmed
from docassist.sampling.std.sampler import StandardSampler
from docassist.models import ModelProfile, ModelBroker, CapabilityVector, Level
from docassist.subjects import AnalysedRepo, PipOutdatedRepo

from typing import NamedTuple, TypedDict

from pydantic_ai.models import Model

class EmbedderConfig(TypedDict):
    model_name: str
    dimension: int | None = None


#fixme this is gonna be a bitch to do properly
repo_dir: Path = Path(__file__).parent.parent
data_dir: Path = repo_dir / "data"

def _level_to_reasoning_effort(reasoning: Level) -> ReasoningEffort:
    return ReasoningEffort.__args__[0].__args__[reasoning.value]

assert _level_to_reasoning_effort(Level.NONE) == "minimal", _level_to_reasoning_effort(Level.NONE)
assert _level_to_reasoning_effort(Level.STRONG) == "high", _level_to_reasoning_effort(Level.STRONG)

class Config(NamedTuple):
    samples_dir: Path = Path(data_dir / "samples")

    fsio: FSIO = SpecializedFSIO(
        YAMLFSIO(),
        [
            (lambda t: issubclass(t, np.ndarray), PickleFSIO())
        ]
    )
    hasher = SHA1OfJsonTrimmed()  # todo this should also be specialized delegate; we dont wanna hash ndarrays as JSON
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

    model_broker: ModelBroker = ModelBroker({
        "openai/gpt-oss-20b": CapabilityVector.defaults(Level.BASIC).override(output_formatting=Level.RELIABLE),
        "openai/gpt-oss-120b": CapabilityVector.defaults(Level.RELIABLE).override(
            output_formatting=Level.STRONG,
            reasoning=Level.STRONG
        ),
        # "deepseek/deepseek-v3.2": CapabilityVector.defaults(Level.STRONG).override(
        #     output_formatting=Level.BASIC,
        #     reasoning=Level.RELIABLE,
        #     hallucination_resistance=Level.RELIABLE
        # ),
        #fixme this should be default=strong with 3 overrides
        "deepseek/deepseek-r1-0528": CapabilityVector.defaults(Level.RELIABLE).override(
            output_formatting=Level.STRONG,
            structured_output=Level.STRONG,
            tool_use=Level.STRONG,
            tool_discipline=Level.STRONG,
            reasoning=Level.STRONG
        ),
        "qwen/qwen3-32b": CapabilityVector.defaults(Level.RELIABLE).override(research=Level.BASIC),
        #todo remove this in the next commit
        # it is here for history, but M2 is too weak and M2.1 is similar to R1
        # if I were to switch from deepsek3.2 to M1, I can consider M2 instead of R1
        "minimax/minimax-m1": CapabilityVector.defaults(Level.RELIABLE).override(reasoning=Level.STRONG),
        "minimax/minimax-m2": CapabilityVector.defaults(Level.BASIC).override(tool_use=Level.RELIABLE),
        "minimax/minimax-m2.1": CapabilityVector.defaults(Level.RELIABLE).override(
            output_formatting=Level.STRONG,
            structured_output=Level.STRONG,
            tool_use=Level.STRONG,
            reasoning=Level.STRONG
        ),
        "text-embedding-3-small": CapabilityVector.embeddings_only()
    })

    model_provider: OpenRouterProvider = OpenRouterProvider(api_key=getenv("OPENAI_API_KEY"))

    def model(self, profile: ModelProfile) -> Model:
        try:
            return self.sampler.over_model(
                OpenAIChatModel(
                    profile.name,
                    provider = self.model_provider,
                    settings=OpenAIChatModelSettings(
                        #fixme this is almost correct - we should base this on agent capability requirements, not advertised model capability
                        # openai_reasoning_effort=_level_to_reasoning_effort(profile.capabilities.reasoning),
                        extra_body={
                            "usage": {
                                "include": True
                            }
                        }
                    )
                )
            )
        except:
            raise

    async_openai: AsyncOpenAI = AsyncOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    embedder_params: dict[str, str | int | None] = {"model_name": "text-embedding-3-small"}

    embedder_config: EmbedderConfig = EmbedderConfig(**embedder_params)
    raw_embedder: Embedder = OpenAIEmbedder(async_openai, **embedder_config)
    embedder: Embedder = sampler.over_embedder(raw_embedder)

    repos: list[AnalysedRepo] = [
        PipOutdatedRepo()
    ]

    indices_dir: str = Path(data_dir / "indices")

CONFIG = Config()