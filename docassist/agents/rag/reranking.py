from pydantic_ai import Agent

from docassist.agents.rag.data import RerankingOutput
from docassist.config import CONFIG
from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask

reranker = Agent(
    name="document reranker",
    model=CONFIG.model,
    output_type=RerankingOutput,
    system_prompt=simple_xml_system_prompt(
        task=PromptingTask(
                context="""
We are retrieving indexed documents for RAG.
We retrieved candidates with redundancy, searching for different queries and getting results that require normalization.
We need to rescore them with more semantic approach than simple index search.
""",
                high_level="Given a list of scored documents and a purpose for current search, provide a list of their updated scores.",
                low_level="""
You will be given full content of the scored documents, as well as their metadata and current score.
You will use that knowledge to produce integer scores in 1-10 range.
You will treat the content as the most important. All the metadata, including previous score, should have much lower priority, but otherwise uniform amongst available details.
You will emit an appropriate output item for each input item. There will be exactly one output item for each input item. Indexes of the documents and their new scores will match.
You will explain your choices, one at the time.
""",
                detailed="""
When choosing the scores for the documents you should use 1-10 scale (without 0, all integers). 
You can use the following to get the gist of what does each value mean.

For example, if the purpose of the search is solving quantum physics problem, then the scores might be:
- a daily newspaper -> 1
- high-school physics book -> 3
- university-level quantum physics book -> 7
- scientific paper explicitly tackling the problem -> 10

If not sure whether to choose a number or one bigger, explain the bigger choice and reevaluate. 
If you choose the lower value at that point, include the original explanation of the bigger choice, explanation of the switch and explanation of the chosen value as if it were chosen originally. 
"""
            ),
        persona="RAG helper specialised in reranking of documents"
    )
)

