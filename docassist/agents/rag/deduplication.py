from docassist.agents.rag.data import DeduplicationOutput, DeduplicationInput
from docassist.structured_agent import StructuredAgent
from docassist.system_prompts import PromptingTask

deduplicator = StructuredAgent(
    name="document deduplicator",
    persona="RAG helper specialised in deduplication of documents",
    input_type=DeduplicationInput,
    output_type=DeduplicationOutput,
    task=PromptingTask(
        context="""
We are retrieving indexed documents for RAG. 
At indexing stage we introduced redundancy and indexed different variants and subsections of each documents.
When retrieving, we found multiple documents stemming from the same original source.
We need to pick one representative of each document.
""",
        high_level="Given a list of document variants and the task they will be used to solve, pick the most relevant and propose a new score.",
        low_level="""
Deduplicate the variants of the retrieved document. 
Reduce the list of candidates to a single, best one. 
Measure 'best' against the given purpose.
Correct (usually raise; if not - include the reason in the explanation) the score to reflect the document being found multiple times. 
Both the input and updated score should be in [0.0, 1.0] range, where 1.0 means the best similarity and 0.0 means no similarity.
Pick the representative by responding with its document ID. You MUST pick EXACTLY one representative and it MUST have the document ID that was present in the input.
Explain your choices, referring to the input data and your inherent knowledge.
""",
        detailed=None
    )
)
