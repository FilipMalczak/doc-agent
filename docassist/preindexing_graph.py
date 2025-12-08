from datetime import datetime, UTC
from dataclasses import dataclass
from io import StringIO
from os.path import join
from sys import prefix
from typing import Any, TextIO, TypedDict, Literal

from uuid import uuid4

import yaml
from pydantic import BaseModel
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append, reduce_list_extend
from yaml import Loader

from docassist.agents.generators.facts_from_file import fact_extractor
from docassist.agents.generators.notes_from_dir import dir_notes_input, SubjectNotes, structurize, \
    dir_note_taker, doc_to_notes
from docassist.agents.generators.notes_from_file import file_note_taker
from docassist.chunkdown import break_to_entries, MarkdownChapter
from docassist.config import CONFIG
from docassist.index.document import SourceMeta, FileNoteMeta, DirNoteMeta, NoteMeta, NoteChunkMeta, FactsMeta, \
    FactsChunkMeta
from docassist.index.protocols import Document, IndexSnapshot
from docassist.index.utils import embed_metadata
from docassist.llmio import object_from_user
from docassist.simple_xml import to_simple_xml
from docassist.subjects import AnalysedRepo, RepoItemType, CodeFilePath

class OK: ...


g = GraphBuilder(name="process-repository", output_type=OK)
sampling = CONFIG.sampler.controller()


@g.step
async def load_sources(ctx: StepContext[None, None, AnalysedRepo]) -> list[Document[SourceMeta]]:
    out = []
    repo = ctx.inputs
    for item_type, source_file in repo.list_all():
        with repo.open(source_file.path, mode="r") as f:
            content = f.read()
        out.append(
            Document(
                id=str(uuid4()),
                content=content,
                metadata=SourceMeta(
                    document_type = "source_file",
                    type = "file",
                    repo_item_type = item_type,
                    path = source_file.path,
                    language = source_file.language
                )
            )
        )
    return out


@g.step
async def take_file_notes(ctx: StepContext[None, None, Document[SourceMeta]]) -> Document[FileNoteMeta]:
    doc = ctx.inputs
    input_data = dict(doc.metadata)
    input_data["content"] = doc.content
    input_ = to_simple_xml(input_data)
    async with sampling.defer_until_success():
        content = (await file_note_taker.run(input_)).output
    return Document(id=str(uuid4()), content=content, metadata=FileNoteMeta(
            document_type = "note",
            subject_path = doc.metadata.path,
            subject_id = doc.id,
            subject_type = "file",
            subject_document_type = "source_file",
            subject_repo_item_type = doc.metadata.repo_item_type,
            subject_language = doc.metadata.language
        )
    )


@g.step
async def take_directory_notes(ctx: StepContext[None, None, list[Document[FileNoteMeta]]]) -> list[Document[DirNoteMeta]]:
    inputs = dir_notes_input(ctx.inputs)
    notes: dict[str, SubjectNotes] = inputs[0]
    paths: list[CodeFilePath] = inputs[1]
    dir_desc = structurize(paths)
    out = []
    for path, files, dirs in dir_desc.depth_first():
        q = object_from_user(
        # q = question(
            sorted(
                [
                    notes[join(path, x)]
                    for x in files + dirs
                ],
                key=lambda n: n.subject_path
            )
        )
        async with sampling.defer_until_success():
            dir_notes = (await dir_note_taker.run(q)).output
        doc = Document(id=str(uuid4()), content=dir_notes, metadata=DirNoteMeta(
                document_type = "note",
                subject_path = path,
                subject_type = "directory"
            )
        )
        out.append(doc)
        n = doc_to_notes(doc)
        notes[path] = n
    return out

@g.step
async def chunk_notes(ctx: StepContext[None, None, Document[NoteMeta]]) -> list[Document[NoteChunkMeta]]:
    doc = ctx.inputs
    return list(break_to_entries(doc))

@g.step
async def extract_facts(ctx: StepContext[None, None, Document[SourceMeta]]) -> Document[FactsMeta]:
    doc = ctx.inputs
    user_msg = dict(doc.metadata)
    user_msg["content"] = doc.content
    async with sampling.defer_until_success():
        facts_obj = (await fact_extractor.run(to_simple_xml(user_msg))).output
    with StringIO() as t:
        yaml.dump(facts_obj, t)
        content = t.getvalue()
    return Document(
        id=str(uuid4()),
        content=content,
        metadata=FactsMeta(
            document_type = "facts",
            subject_id = doc.id,
            subject_path = doc.metadata.path,
            subject_type = "file",
            subject_document_type = "source_file",
            subject_repo_item_type = doc.metadata.repo_item_type,
            subject_language = doc.metadata.language
    )
    )

@g.step
async def chunk_facts(ctx: StepContext[None, None, Document[FactsMeta]]) -> list[Document[FactsChunkMeta]]:
    doc = ctx.inputs
    facts = yaml.load(doc.content, Loader=Loader)
    def i():
        for i, f in enumerate(facts.facts):
            yield Document(
                id=str(uuid4()),
                content=f.fact,
                metadata=FactsChunkMeta(
                    document_type = "chunk",
                    chunk_source_document_type = "facts",
                    chunk_variant = "simple",
                    chunk_coordinates = (i, ),
                    chunked_facts_id = doc.id,
                    chunked_facts_subject_path = doc.metadata.subject_path,
                    chunked_facts_subject_type = doc.metadata.subject_type
                )
            )
            with StringIO() as t:
                yaml.dump(f, t)
                single_explained = t.getvalue()
            yield Document(
                id=str(uuid4()),
                content=single_explained,
                metadata=FactsChunkMeta(
                    document_type = "chunk",
                    chunk_source_document_type = "facts",
                    chunk_variant = "explained",
                    chunk_coordinates = (i,),
                    chunked_facts_id = doc.id,
                    chunked_facts_subject_path = doc.metadata.subject_path,
                    chunked_facts_subject_type = doc.metadata.subject_type
                )
            )
    return list(i())


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