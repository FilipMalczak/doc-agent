from logging import getLogger
from pathlib import Path
from pprint import pprint

from logfire import span, instrument
from namesgenerator import get_random_name

from docassist.config import CONFIG
from docassist.index.faiss import FAISSIndex
from docassist.preindexing_graph import repo_preindexing
from docassist.structure.materialize.materializer import Materializer
from docassist.structure.spec import root_specification
from docassist.subjects import AnalysedRepo

log = getLogger(__name__)

@instrument()
async def handle_repo(repo: AnalysedRepo):
    repo_index_path = Path(CONFIG.indices_dir / repo.name)
    index_ready_marker = repo_index_path / "index_ready.bool"
    if index_ready_marker.exists():
        log.info(f"Index for repository {repo.name} already exists at {index_ready_marker}; loading!")
        index = await FAISSIndex.load(repo_index_path, CONFIG.embedder)
        log.info(f"Index for {repo.name} loaded")
    else:
        index = FAISSIndex(CONFIG.embedder)
        docs = await repo_preindexing(repo)
        await index.add(docs)
        await index.store(repo_index_path)
        index_ready_marker.touch(exist_ok=False)
    materializer = Materializer(index=index, sampling=CONFIG.sampler.controller())
    pprint(await materializer.materialize_specification(root_specification))





async def main():
    session_name = get_random_name()
    print("Session name:", session_name)
    with span("session") as s:
        s.set_attribute("session.name", session_name)
        for repo in CONFIG.repos:
            await handle_repo(repo)