from datetime import datetime, UTC
from dataclasses import dataclass
from sys import prefix
from typing import Any

from uuid import uuid4
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from docassist.agents.note_taker import Markdown, note_taker
from docassist.config import CONFIG
from docassist.index.protocols import Document, IndexSnapshot
from docassist.simple_xml import to_simple_xml
from docassist.subjects import AnalysedRepo, RepoItemType, CodeFilePath


@dataclass
class State:
    ...

@dataclass
class OK:
    ...

def embed_metadata(document: Document, prefix: str | None, fields: list[str]) -> dict[str, Any]:
    return {
        prefix+"_"+k if prefix is not None else k: v
        for k, v in document.metadata.items()
    }

g = GraphBuilder(state_type=State, output_type=OK)

@g.step
async def load_sources(ctx: StepContext[State, None, AnalysedRepo]) -> list[Document]:
    out = []
    repo = ctx.inputs
    for item_type, source_file in repo.list_all():
        with repo.open(source_file.path, mode="r") as f:
            content = f.read()
        out.append(
            Document(
                id=str(uuid4()),
                content=content,
                metadata={ # fixme we need proper tools for metadata filtering in rag
                    "document_type": "source_file",
                    "repo_item_type": item_type,
                    "path": source_file.path,
                    "language": source_file.language
                }
            )
        )
    return out

@g.step
async def take_notes(ctx: StepContext[State, None, Document]) -> Document:
    doc = ctx.inputs
    input_data = dict(doc.metadata)
    input_data["content"] = doc.content
    input_ = to_simple_xml(input_data)
    content = (await note_taker.run(input_)).output
    return Document(id=str(uuid4()), content=content, metadata={
        "document_type": "note",
        **embed_metadata(doc, "subject", ["id", "path", "document_type"])
    })

@g.step
async def build_index(ctx: StepContext[State, None, list[Document]]) -> IndexSnapshot:
    # take empty index (from config), populate it; not the best way to do this
    idx = CONFIG.index
    await idx.add(ctx.inputs)
    return IndexSnapshot(path=f"./indices/index_{datetime.now(UTC).isoformat()}", index=idx)

@g.step
async def ack(ctx: StepContext[State, None, IndexSnapshot]) -> OK:
    return OK()

collect_docs = g.join(reduce_list_append, initial=[]) #fixme

g.add(
    g.edge_from(g.start_node).to(load_sources),
    g.edge_from(load_sources).map().to(collect_docs),
    g.edge_from(load_sources).map().to(take_notes),
    g.edge_from(take_notes).to(collect_docs),
    g.edge_from(collect_docs).to(build_index),
    g.edge_from(build_index).to(ack),
    g.edge_from(ack).to(g.end_node)
)

GRAPH = g.build()