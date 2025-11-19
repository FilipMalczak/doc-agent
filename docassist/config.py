from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from docassist.index.faiss import FAISSIndex
from docassist.index.openai_embedder import OpenAIEmbedder
from docassist.index.protocols import Embedder, DocumentIndex

load_dotenv()
from os import getenv
from typing import NamedTuple, Self, TypedDict

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from docassist.sampling.usual import get_sampler
from docassist.subjects import AnalysedRepo, PipOutdatedRepo

class EmbedderConfig(TypedDict):
    model: str
    dimension: int | None = None

class Config(NamedTuple):
    raw_model: Model = OpenAIChatModel(
        'openai/gpt-oss-120b',
        provider=OpenRouterProvider(api_key=getenv("OPENAI_API_KEY")),
    )
    embedder_params: dict[str, str | int | None] = {"model": "text-embedding-3-small"}

    model: Model = get_sampler().over_model(raw_model)
    repos: list[AnalysedRepo] = [
        PipOutdatedRepo()
    ]
    async_openai: AsyncOpenAI = raw_model.client
    embedder_config: EmbedderConfig = EmbedderConfig(**embedder_params)
    embedder: Embedder = OpenAIEmbedder(async_openai, **embedder_config)
    index: DocumentIndex = FAISSIndex(embedder)


CONFIG = Config()