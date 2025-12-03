from pydantic import BaseModel
from pydantic_ai import Agent

from docassist.agents.rag.data import DeduplicationOutput
from docassist.config import CONFIG
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask


deduplicator = Agent(
    name="document deduplicator",
    model=CONFIG.model,
    output_type=DeduplicationOutput,
    system_prompt=simple_xml_system_prompt(
        task=PromptingTask(
            context="""
We are retrieving indexed documents for RAG. 
At indexing stage we introduced redundancy and indexed different variants and subsections of each documents.
When retrieving, we found multiple documents stemming from the same original source.
We need to pick one representative of each document.
""",
            high_level="Given a list of document variants and the task they will be used to solve, pick the most relevant and propose a new score.",
            low_level="""
Deduplicate the variants of the retrieved task. 
Reduce the list of candidates to a single, best one. 
Measure 'best' against the given task. 
Correct (usually raise; if not - include the reason in the explanation) the score to reflect the document being found multiple times. 
Explain your choices.
""",
            detailed=None
        ),
        persona="RAG helper specialised in deduplication of documents"
    )
)

