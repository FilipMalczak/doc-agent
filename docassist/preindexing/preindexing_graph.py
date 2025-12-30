from functools import wraps
from io import StringIO
from os.path import join
from typing import Awaitable, Callable, Any
from uuid import uuid4

import yaml
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append, reduce_list_extend
from yaml import Loader

from docassist.chunkdown import break_to_entries
from docassist.config import CONFIG
from docassist.index.document import SourceMeta, FileNoteMeta, DirNoteMeta, NoteMeta, NoteChunkMeta, FactsMeta, \
    FactsChunkMeta
from docassist.index.protocols import Document
from docassist.preindexing.agents.notes_from_file import file_note_taker, FullyDescribedFile
from docassist.structured_agent import call_agent, StructuredAgent
from docassist.subjects import AnalysedRepo, CodeFilePath

g = GraphBuilder(name="process-repository", output_type=list[Document])
sampling = CONFIG.sampler.controller()

#fixme this belongs to the other graph, where the agents are composed into full pipeline
def agent_step[I, O](a: StructuredAgent[I, O]):
    def decorator(foo: Callable[[StepContext[Any, Any, I]], O | Awaitable[O]]) -> Callable[[StepContext[Any, Any, I]], Awaitable[O]]:
        @g.step
        @wraps(foo)
        def impl(ctx: StepContext[Any, Any, I]) -> Awaitable[O]:
            return a.run(ctx.inputs)
        return impl
    return decorator


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
    agent = file_note_taker.parametrized_as(role="end-user", relationship_to_project="developer")
    input_data = FullyDescribedFile.of(ctx.inputs)
    content = await agent.run(input_data)
    return Document(id=str(uuid4()), content=content, metadata=FileNoteMeta(
            document_type = "note",
            subject_path = doc.metadata.path,
            subject_id = doc.id,
            subject_type = "file",
            subject_document_type = "source_file",
            subject_repo_item_type = doc.metadata.repo_item_type,
            subject_language = doc.metadata.languaged
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
        q = sorted([notes[join(path, x)] for x in files + dirs], key=lambda n: n.subject_path)
        dir_notes = await dir_note_taker.run(q)
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
    facts_obj = await fact_extractor.run(user_msg)
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
    g.edge_from(all_docs).to(g.end_node)
)

repo_preindexing_graph = g.build()

def repo_preindexing(repo: AnalysedRepo) -> Awaitable[list[Document]]:
    return repo_preindexing_graph.run(inputs=repo)

# print(repo_preindexing.render())