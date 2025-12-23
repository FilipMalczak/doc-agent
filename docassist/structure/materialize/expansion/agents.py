from docassist.structure.materialize.expansion.models import DomainInterpretationInput, InterpretedDomain, \
    CandidateHypothesis, EntityAnchorInput, EntityAnchorOutput, SchemaProjectorInput, SchemaProjectedItem, \
    AnchoredEntity
from docassist.structured_agent import StructuredAgent
from docassist.system_prompts import PromptingTask

domain_interpreter = StructuredAgent(
    name="domain interpreter",
    persona="expert in translating natural language specifications into formal constraints",
    turbo=True,
    task=PromptingTask(
        context=(
            "You are part of a documentation materialization pipeline. "
            "Your role is to interpret a natural-language domain description "
            "into explicit, structured constraints without assuming project contents."
        ),
        high_level=(
            "Translate the given domain description into a normalized representation "
            "that downstream agents can rely on to identify relevant entities."
        ),
        low_level=(
            "Do not enumerate examples or entities. Do not infer existence. "
            "Focus on identifying the kind of things requested, their qualifiers, "
            "and any explicit or implicit cardinality constraints."
        ),
        detailed=(
            "Treat ambiguous language conservatively. If a term like 'basic', "
            "'common', or 'public' appears, preserve it as a qualifier instead "
            "of interpreting it. If cardinality is unclear, leave it unset."
        ),
    ),
    input_type=DomainInterpretationInput,
    output_type=InterpretedDomain
)

candidate_hypothesizer = StructuredAgent(
    name="candidate hypothesizer",
    persona="analyst generating plausible entity hypotheses from specifications",
    turbo=True,
    task=PromptingTask(
        context=(
            "You assist in discovering project entities by proposing candidate "
            "descriptions derived from a formalized domain request."
        ),
        high_level=(
            "Generate a set of plausible candidate descriptions that could correspond "
            "to real entities satisfying the domain constraints."
        ),
        low_level=(
            "Optimize for recall, not precision. Candidates are not claims of existence. "
            "Do not name concrete entities as facts. Provide retrieval-oriented hints."
        ),
        detailed=(
            "Each candidate should be described abstractly and accompanied by keywords "
            "or phrases suitable for knowledge base search. Generate more candidates "
            "than the requested cardinality if necessary."
        ),
    ),
    input_type=InterpretedDomain,
    output_type=list[CandidateHypothesis]
)

entity_anchor = StructuredAgent(
    name="entity anchor",
    persona="knowledge base specialist responsible for grounding entities",
    turbo=True,
    task=PromptingTask(
        context=(
            "You are responsible for resolving candidate hypotheses into real, "
            "knowledge-base-backed entities."
        ),
        high_level=(
            "Identify which candidates correspond to actual project entities by "
            "searching and analyzing the knowledge base."
        ),
        low_level=(
            "Only emit entities that are supported by clear knowledge base evidence. "
            "Discard candidates that cannot be grounded. Enforce cardinality constraints "
            "strictly if specified."
        ),
        detailed=(
            "Use the search tool extensively. Deduplicate similar results. "
            "For each accepted entity, provide identifiers, evidence document references, "
            "and a confidence score reflecting strength of support."
        ),
    ),
    input_type=EntityAnchorInput,
    output_type=list[AnchoredEntity],
)

schema_projector = StructuredAgent(
    name="schema projector",
    persona="technical writer specializing in schema-driven content projection",
    turbo=True,
    task=PromptingTask(
        context=(
            "You transform grounded project entities into structured documentation "
            "entries according to a provided variable schema."
        ),
        high_level=(
            "For each anchored entity, populate the requested variables using only "
            "knowledge base evidence and constrained summarization."
        ),
        low_level=(
            "Do not introduce new facts. Do not reinterpret entity meaning. "
            "If a variable cannot be populated from evidence, leave it empty or note it."
        ),
        detailed=(
            "Treat variable descriptions as semantic guidance. "
            "Preserve traceability by associating populated values with evidence sources."
        ),
    ),
    input_type=SchemaProjectorInput,
    output_type=SchemaProjectedItem,
)
