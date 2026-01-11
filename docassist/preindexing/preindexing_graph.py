from functools import wraps
from io import StringIO
from typing import Awaitable, Callable, Any

import yaml
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append, reduce_list_extend
from pydantic_graph.beta.paths import EdgePath
from yaml import Loader

from docassist.chunkdown import break_to_entries
from docassist.config import CONFIG
from docassist.index.protocols import Document
from docassist.parametrized import Parametrized
from docassist.preindexing.agents.facts_from_file import fact_extractor, Facts
from docassist.preindexing.agents.notes_from_file import file_note_taker
from docassist.preindexing.perspectives import PERSPECTIVES, AudienceRole, AudienceToProjectRelationship, \
    PerspectivePointer
from docassist.structured_agent import StructuredAgent
from docassist.subjects import AnalysedRepo

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

def parametrized_step[**P, I, O](
        foo: Callable[[StepContext[Any, Any, I], P.kwargs], Awaitable[O]]
) -> Parametrized[Callable[[StepContext[Any, Any, I]], Awaitable[O]]]:
    def make_step(name_suffix: str, params: P.kwargs) -> Callable[[StepContext[Any, Any, I]], Awaitable[O]]:
        def impl(ctx: StepContext[Any, Any, I]) -> Awaitable[O]:
            return foo(ctx, **params)
        impl.__name__ = foo.__name__ + name_suffix
        out = g.step(impl)
        return out
    return Parametrized(PERSPECTIVES, make_step)

SourceDocument = Document

@g.step
async def load_sources(ctx: StepContext[None, None, AnalysedRepo]) -> list[SourceDocument]:
    out = []
    repo = ctx.inputs
    for repo_function, source_file in repo.list_all():
        with repo.open(source_file.path, mode="r") as f:
            content = f.read()
        doc = Document.source_file(
            content=content,
            path=source_file.path,
            language=source_file.language,
            repo_function=repo_function,
        )
        out.append(doc)
    return out

NoteDocument = Document

@parametrized_step
async def take_file_notes(ctx: StepContext[None, None, SourceDocument], *,
                          role: AudienceRole, relationship_to_project: AudienceToProjectRelationship
                          ) -> list[NoteDocument]:
    doc = ctx.inputs
    agent = file_note_taker.parametrized_by(role=role, relationship_to_project=relationship_to_project)
    content = await agent.run(doc)
    if content is None:
        return []
    return [
        doc.derive_note(
            content=content,
            perspective=PerspectivePointer(role=role, relationship_to_project=relationship_to_project)
        )
    ]


NoteDocument = Document
NoteChapterDocument = Document

@g.step
async def chunk_notes(ctx: StepContext[None, None, NoteDocument]) -> list[NoteChapterDocument]:
    doc = ctx.inputs
    return list(break_to_entries(doc))

FactsDocument = Document
SingleFactDocument = Document

@parametrized_step
async def extract_facts(ctx: StepContext[None, None, SourceDocument], *,
                        role: AudienceRole, relationship_to_project: AudienceToProjectRelationship
                        ) -> FactsDocument:
    doc = ctx.inputs
    agent = fact_extractor.parametrized_by(role=role, relationship_to_project=relationship_to_project)
    facts_obj: Facts | None = await agent.run(doc)
    if facts_obj is None:
        return []
    with StringIO() as t:
        yaml.dump(facts_obj.model_dump(mode="json"), t)
        content = t.getvalue()
    return [
        doc.derive_facts(
            content=content,
            perspective=PerspectivePointer(role=role, relationship_to_project=relationship_to_project)
        )
    ]

@g.step
async def chunk_facts(ctx: StepContext[None, None, FactsDocument]) -> list[SingleFactDocument]:
    doc = ctx.inputs
    facts = Facts(**yaml.load(doc.content, Loader=Loader))
    def i():
        for i, f in enumerate(facts.facts):
            yield doc.derive_fact(
                content=f.fact,
                index=i,
                explained=False
            )
            with StringIO() as t:
                yaml.dump(f, t)
                single_explained = t.getvalue()
            yield doc.derive_fact(
                content=single_explained,
                index=i,
                explained=True
            )
    return list(i())


all_notes = g.join(reduce_list_extend, initial=[])
all_facts = g.join(reduce_list_extend, initial=[])
all_chunks = g.join(reduce_list_extend, initial=[])
all_docs = g.join(reduce_list_extend, initial=[])

g.add(
    g.edge_from(g.start_node).to(load_sources),
    g.edge_from(load_sources).to(all_docs),
)

for p in PERSPECTIVES:
    take_n = take_file_notes.parametrized_by(**p)
    extract_f = extract_facts.parametrized_by(**p)
    g.add(
        g.edge_from(load_sources).map().to(take_n),
        g.edge_from(take_n).to(all_notes),
        g.edge_from(load_sources).map().to(extract_f),
        g.edge_from(extract_f).to(all_facts)
    )

g.add( #todo add "drop meaningless" filters
    g.edge_from(all_notes).to(all_docs),
    g.edge_from(all_notes).map().to(chunk_notes),
    g.edge_from(chunk_notes).to(all_chunks),
    g.edge_from(all_chunks).to(all_docs),
    g.edge_from(all_facts).to(all_docs),
    g.edge_from(all_facts).map().to(chunk_facts),
    g.edge_from(chunk_facts).to(all_chunks),
    g.edge_from(all_docs).to(g.end_node)
)

repo_preindexing_graph = g.build()

def repo_preindexing(repo: AnalysedRepo) -> Awaitable[list[Document]]:
    return repo_preindexing_graph.run(inputs=repo)

# print(repo_preindexing_graph.render())