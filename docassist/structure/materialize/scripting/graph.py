from typing import Awaitable, Callable

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from docassist.index.document import Document
from docassist.structure.materialize.expansion.agents import domain_interpreter, candidate_hypothesizer, entity_anchor, \
    schema_projector
from docassist.structure.materialize.expansion.models import SchemaProjectedItem, InterpretedDomain, \
    DomainInterpretationInput, CandidateHypothesis, AnchoredEntity, SchemaVariable, SchemaProjectorInput
from docassist.structure.materialize.scripting.agents import planner, scribe
from docassist.structure.materialize.scripting.models import ResearchSubject, ResearchResults, ManuscriptPlan
from docassist.structured_agent import StructuredAgent

g = GraphBuilder(name="prepare-manuscript", output_type=list[SchemaProjectedItem])

DocumentId = str
DocumentLoader = Callable[[DocumentId], Document]


@g.step
async def research_topic(ctx: StepContext[None, DocumentLoader, ResearchSubject]) -> Awaitable[tuple[ResearchSubject, ResearchResults]] :
    results = await planner.run(ctx.inputs)
    return (ctx.inputs, results)

@g.step
async def hydrate_plan(ctx: StepContext[None, DocumentLoader, tuple[ResearchSubject, ResearchResults]]) -> Awaitable[ManuscriptPlan]:
    subject, results = ctx.inputs
    loader = ctx.deps
    return ManuscriptPlan(
        subject=subject,
        plan_as_text=results.plan_as_text,
        notes=results.notes,
        knowledge=[
            loader(evidence_id)
            for evidence_id in results.evidence_ids
        ]
    )

@g.step
def write_on_subject(ctx: StepContext[None, DocumentLoader, ManuscriptPlan]) -> Awaitable[str]:
    return scribe.run(ctx.inputs)

g.add(
    g.edge_from(g.start_node).to(research_topic),
    g.edge_from(research_topic).to(hydrate_plan),
    g.edge_from(hydrate_plan).to(write_on_subject),
    g.edge_from(write_on_subject).to(g.end_node)
)

prepare_manuscript_graph = g.build()

#todo params of this function
def prepare_manuscript(domain_description: str, variables: dict[str, str]) -> Awaitable[str]:
    return prepare_manuscript_graph.run(
        inputs=DomainInterpretationInput(domain_description=domain_description),
        deps=[ SchemaVariable(name=k, description=v) for k, v in variables.items() ]
    )
