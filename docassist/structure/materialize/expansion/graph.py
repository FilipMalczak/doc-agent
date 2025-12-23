from typing import Awaitable

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from docassist.structure.materialize.expansion.agents import domain_interpreter, candidate_hypothesizer, entity_anchor, \
    schema_projector
from docassist.structure.materialize.expansion.models import SchemaProjectedItem, InterpretedDomain, \
    DomainInterpretationInput, CandidateHypothesis, AnchoredEntity, SchemaVariable, SchemaProjectorInput
from docassist.structured_agent import StructuredAgent

g = GraphBuilder(name="expand-domain", output_type=list[SchemaProjectedItem])


@g.step
def interpret_domain(ctx: StepContext[None, list[SchemaVariable], DomainInterpretationInput]) -> Awaitable[InterpretedDomain] :
    return domain_interpreter.run(ctx.inputs)

@g.step
def hypothesize_candidates(ctx: StepContext[None, list[SchemaVariable], InterpretedDomain]) -> Awaitable[list[CandidateHypothesis]]:
    return candidate_hypothesizer.run(ctx.inputs)

@g.step
def anchor_entites(ctx: StepContext[None, list[SchemaVariable], list[CandidateHypothesis]]) -> Awaitable[list[AnchoredEntity]]:
    return entity_anchor.run(ctx.inputs)

@g.step
def project_schema(ctx: StepContext[None, list[SchemaVariable], AnchoredEntity]) -> Awaitable[SchemaProjectedItem]:
    return schema_projector.run(SchemaProjectorInput(entity=ctx.inputs, variables=ctx.deps))

results = g.join(reduce_list_append, initial=[])

g.add(
    g.edge_from(g.start_node).to(interpret_domain),
    g.edge_from(interpret_domain).to(hypothesize_candidates),
    g.edge_from(hypothesize_candidates).to(anchor_entites),
    g.edge_from(anchor_entites).map().to(project_schema),
    g.edge_from(project_schema).to(results),
    g.edge_from(results).to(g.end_node)
)

domain_expanding_graph = g.build()

def expand_domain(domain_description: str, variables: dict[str, str]) -> Awaitable[list[SchemaProjectedItem]]:
    return domain_expanding_graph.run(
        inputs=DomainInterpretationInput(domain_description=domain_description),
        deps=[ SchemaVariable(name=k, description=v) for k, v in variables.items() ]
    )
