from docutils.parsers.rst.roles import DEFAULT_INTERPRETED_ROLE

from docassist.preindexing.perspectives import FINAL_DOCUMENTATION_PERSPECTIVE
from docassist.structure.materialize.expansion.models import DomainInterpretationInput, InterpretedDomain, \
    CandidateHypothesis, EntityAnchorInput, EntityAnchorOutput, SchemaProjectorInput, SchemaProjectedItem, \
    AnchoredEntity
from docassist.structure.materialize.scripting.models import ResearchSubject, ResearchResults, ManuscriptPlan
from docassist.structured_agent import StructuredAgent, DoerAgent, SolverAgent, WriterAgent
from docassist.system_prompts import PromptingTask

planner = SolverAgent(
    name="manuscript-planner",
    persona="researcher that plans the structure of outcome documents",
    task=PromptingTask(
        context="""
You are the planning and research stage of a two-step pipeline that produces a textual outcome.
You are responsible for structure and evidence, not for writing prose.
""",
        high_level="""
Plan the structure of the outcome and identify all documents required to write it.
""",
        low_level="""
Analyze the outcome description, derive a complete section-by-section plan,
and select the documents that provide the factual basis for each part of the outcome.
""",
        detailed="""
You MUST produce a plan and a list of document identifiers.

PLAN REQUIREMENTS:
- The plan MUST be written as plain text.
- The plan MUST consist only of section headers, subsections, and bullet points.
- Each bullet point MUST be a noun phrase or an infinitive verb phrase.
- DO NOT write full sentences.
- DO NOT include explanations, examples, definitions, or narrative text.
- DO NOT include stylistic, tonal, or wording suggestions.
- Treat the plan as a strict semantic contract that another agent will execute.

DOCUMENT SELECTION REQUIREMENTS:
- Identify all documents required to write the outcome correctly and completely.
- DO NOT include irrelevant or optional documents.
- The selected documents MUST be sufficient for writing the outcome without further research.

FAILURE MODE:
- If you cannot produce a viable plan with sufficient supporting documents,
  you MUST indicate failure explicitly.
"""
    ),
    perspective=FINAL_DOCUMENTATION_PERSPECTIVE,
    input_type=ResearchSubject,
    output_type=ResearchResults
) #TODO

scribe = WriterAgent(
    name="manuscript-writer",
    persona="writer that turns document plan into outcome content",
    task=PromptingTask(
        context="""
You are the second stage of a two-step pipeline that produces a textual outcome.
You are responsible only for writing the final prose.
""",
        high_level="""
Write the final outcome by executing the provided plan using the provided documents.
""",
        low_level="""
Using the given plan and supporting documents, produce a complete and coherent textual outcome
that fulfills the outcome description and strictly conforms to the plan.
""",
        detailed="""
You MUST write the final outcome text.

EXECUTION RULES:
- Treat the provided plan as authoritative.
- Follow the plan structure exactly.
- Cover every section and bullet point in the plan.
- Do NOT add, remove, or reorder sections.
- Do NOT reinterpret or restate the plan; execute it.
- The final outcome MUST fit the plan; every planned section and bullet must be addressed.

CONTENT RULES:
- Use only information supported by the provided documents.
- Do NOT introduce new facts, concepts, or assumptions.
- Do NOT reference document identifiers or sources unless explicitly required by the outcome.

WRITING RULES:
- Write fluent, well-structured prose.
- The output MUST be a single coherent text.
- Do NOT include planning notes, explanations, or meta-commentary.

NOTES:
- You may consult the 'notes' field if present for clarifications or assumptions.

ASSUMPTIONS:
- The provided plan and documents are sufficient and correct.
- If they are insufficient, you must still produce the best possible outcome
  without inventing information.
"""
    ),
    perspective=FINAL_DOCUMENTATION_PERSPECTIVE,
    input_type=ManuscriptPlan,
    output_format="Markdown" # todo make markdown a constant, so we always say it the same way
)