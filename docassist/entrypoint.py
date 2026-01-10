from io import StringIO
from logging import getLogger
from pathlib import Path
from pprint import pprint
from typing import TypedDict

import yaml
from logfire import span
from namesgenerator import get_random_name

from docassist.config import CONFIG
from docassist.index.faiss import FAISSIndex
from docassist.preindexing.preindexing_graph import repo_preindexing
from docassist.retries import step
from docassist.structure.materialize.materializer import Materializer
from docassist.structure.spec import root_specification
from docassist.subjects import AnalysedRepo
from docassist.usage import USAGE_COLLECTOR

log = getLogger(__name__)

@step
async def handle_repo(repo: AnalysedRepo):
    repo_index_path = Path(CONFIG.indices_dir / repo.name)
    index_ready_marker = repo_index_path / "index_ready.bool"
    if index_ready_marker.exists():
        log.info(f"Index for repository {repo.name} already exists at {index_ready_marker}; loading!")
        with span("load index"):
            index = await FAISSIndex.load(repo_index_path, CONFIG.embedder)
        log.info(f"Index for {repo.name} loaded")
    else:
        index = FAISSIndex(CONFIG.embedder)
        docs = await repo_preindexing(repo)
        with span("make index"):
            with span("add"):
                await index.add(docs)
            with span("store"):
                await index.store(repo_index_path)
                index_ready_marker.touch(exist_ok=False)
    materializer = Materializer(index=index, sampling=CONFIG.sampler.controller())
    pprint(await materializer.materialize_specification(root_specification))

class ModelCosts(TypedDict):
    input_per_m: float
    output_per_m: float

def calculate(costs: ModelCosts, input: int, output: int) -> float:
    return input*costs["input_per_m"]/1000000.0 + output*costs["output_per_m"]/1000000.0

COSTS = {
    'deepseek/deepseek-r1-0528': {"input_per_m": 0.4, "output_per_m": 1.75},
    'deepseek/deepseek-v3.2': {"input_per_m": 0.27, "output_per_m": 0.4},
    'openai/gpt-oss-120b': {"input_per_m": 0.039, "output_per_m": 0.19},
    'openai/gpt-oss-20b': {"input_per_m": 0.02, "output_per_m": 0.1},
    'qwen/qwen3-32b': {"input_per_m": 0.4, "output_per_m": 0.8}, #costs for the most expensive provider; for most of the calls its much lower
    'text-embedding-3-small': {"input_per_m": 0.02, "output_per_m": 0.0},
}

async def main():
    session_name = get_random_name()
    print("Session name:", session_name)
    with span("session") as s:
        s.set_attribute("session.name", session_name)
        try:
            for repo in CONFIG.repos:
                await handle_repo(repo)
        finally:
            details = {k: dict(v) for k, v in USAGE_COLLECTOR.by_model.items()}
            print("USAGE")
            print(details)
            def calc(usage_class):
                out = {
                    model_name: calculate(
                        COSTS[model_name],
                        USAGE_COLLECTOR.by_model[usage_class][model_name]["prompt_tokens"],
                        USAGE_COLLECTOR.by_model[usage_class][model_name]["completion_tokens"]
                    )
                    for model_name in USAGE_COLLECTOR.by_model[usage_class]
                }
                out["total"] = sum(out.values())
                return out

            spending = {
                "would_spend": calc("all"),
                "spent_this_time": calc("unsampled"),
                "saved_this_time": calc("sampled"),
            }
            print("SPENDING")
            print(spending)