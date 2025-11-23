from datetime import datetime, UTC
from dataclasses import dataclass
from os.path import join
from sys import prefix
from typing import Any

from uuid import uuid4

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append, reduce_list_extend

from docassist.agents.generators.notes_from_dir import dir_notes_input, SubjectNotes, structurize, question, \
    dir_note_taker, doc_to_notes
from docassist.agents.generators.notes_from_file import file_note_taker
from docassist.chunkdown import break_to_entries, MarkdownChapter
from docassist.config import CONFIG
from docassist.index.protocols import Document, IndexSnapshot
from docassist.index.utils import embed_metadata
from docassist.simple_xml import to_simple_xml
from docassist.subjects import AnalysedRepo, RepoItemType, CodeFilePath

class OK: ...


g = GraphBuilder(name="process-repository", output_type=OK)


@g.step
async def load_sources(ctx: StepContext[None, None, AnalysedRepo]) -> list[Document]:
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
                    "type": "file",
                    "document_type": "source_file",
                    "repo_item_type": item_type,
                    "path": source_file.path,
                    "language": source_file.language
                }
            )
        )
    return out

@g.step
async def take_file_notes(ctx: StepContext[None, None, Document]) -> Document:
    doc = ctx.inputs
    input_data = dict(doc.metadata)
    input_data["content"] = doc.content
    input_ = to_simple_xml(input_data)
    content = (await file_note_taker.run(input_)).output
    return Document(id=str(uuid4()), content=content, metadata={
        "document_type": "note",
        "subject_id": doc.id,
        **embed_metadata(doc.metadata, "subject")
    })

@g.step
async def take_directory_notes(ctx: StepContext[None, None, list[Document]]) -> list[Document]:
    inputs = dir_notes_input(ctx.inputs)
    notes: dict[str, SubjectNotes] = inputs[0]
    paths: list[CodeFilePath] = inputs[1]
    dir_desc = structurize(paths)
    out = []
    for path, files, dirs in dir_desc.depth_first():
        q = question(
            sorted(
                [
                    notes[join(path, x)]
                    for x in files + dirs
                ],
                key=lambda n: n.subject_path
            )
        )
        dir_notes = (await dir_note_taker.run(q)).output
        doc = Document(id=str(uuid4()), content=dir_notes, metadata={
            "document_type": "note",
            "subject_path": path,
            "subject_type": "directory"
        })
        out.append(doc)
        n = doc_to_notes(doc)
        notes[path] = n
    return out

def d(x: str | None = None):
    x = x or "..."
    return Document(id=x, content=x, metadata={})

@g.step
async def chunk_notes(ctx: StepContext[None, None, Document]) -> list[Document]:
    doc = ctx.inputs
    return list(break_to_entries(doc))

@g.step
async def extract_facts(ctx: StepContext[None, None, Document]) -> Document:
    return d("extract_facts")

@g.step
async def chunk_facts(ctx: StepContext[None, None, Document]) -> list[Document]:
    return [d("chunk_facts")]

@g.step
async def build_index(ctx: StepContext[None, None, list[Document]]) -> IndexSnapshot:
    # take empty index (from config), populate it; not the best way to do this
    idx = CONFIG.index
    await idx.add(ctx.inputs)
    return IndexSnapshot(path=f"./indices/index_{datetime.now(UTC).isoformat()}", index=idx)

@g.step
async def ack(ctx: StepContext[None, None, IndexSnapshot]) -> OK:
    results = await ctx.inputs.index.query(["project name", "project_name", "name of the project"], total_results=5)
    for r in results:
        print(r)
    return OK()

all_file_notes = g.join(reduce_list_append, initial=[])
all_notes = g.join(reduce_list_extend, initial=[])
all_facts = g.join(reduce_list_append, initial=[])
all_chunks = g.join(reduce_list_extend, initial=[])
all_docs = g.join(reduce_list_extend, initial=[])

#fixme missing facts pipeline
g.add(
    g.edge_from(g.start_node).to(load_sources),
    g.edge_from(load_sources).to(all_docs),
    g.edge_from(load_sources).map().to(take_file_notes),
    g.edge_from(load_sources).map().to(extract_facts),
    g.edge_from(take_file_notes).to(all_file_notes),
    g.edge_from(all_file_notes).to(take_directory_notes),
    g.edge_from(take_directory_notes).to(all_notes),
    g.edge_from(all_file_notes).to(all_notes),
    g.edge_from(all_notes).to(all_docs),
    g.edge_from(all_notes).map().to(chunk_notes),
    g.edge_from(chunk_notes).to(all_chunks),
    g.edge_from(all_chunks).to(all_docs),
    g.edge_from(extract_facts).to(all_facts),
    g.edge_from(all_facts).to(all_docs),
    g.edge_from(extract_facts).to(chunk_facts),
    g.edge_from(chunk_facts).to(all_chunks),
    g.edge_from(all_docs).to(build_index),
    g.edge_from(build_index).to(ack),
    g.edge_from(ack).to(g.end_node)
)

process_directory = g.build()
print(process_directory.render())
# GRAPH.run = trace()(GRAPH.run)