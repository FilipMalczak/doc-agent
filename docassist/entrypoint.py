

from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext

from docassist.agents.note_taker import SourceFile, take_notes
from docassist.config import CONFIG
from docassist.graph import GRAPH


async def main():
    for repo in CONFIG.repos:
        await GRAPH.run(inputs=repo)
        # for src_t, src_p in repo.list_all():
            # with repo.open(src_p.path, mode="r") as f:
            #     content = f.read()
            # inp = SourceFile(src_p.path, src_p.language, src_t, content)
            # md = await take_notes(inp)
            # print("Notes for", inp)
            # print(md)
            # print("="*80)