from docassist.agents.rag.data import RerankedItem, RerankingInput, RerankingOutput
from docassist.parametrized import Parametrized
from docassist.preindexing.perspectives import PERSPECTIVES, perspective
from docassist.structured_agent import StructuredAgent, DoerAgent, SolverAgent
from docassist.system_prompts import PromptingTask



reranker = Parametrized(
    parameters=PERSPECTIVES,
    factory=lambda name_suffix, params:
        DoerAgent(
            name="document reranker",
            persona="RAG helper specialised in reranking of documents",
            perspective=perspective(**params),
            task=PromptingTask(
                context="""
We are retrieving indexed documents for RAG.
We retrieved candidates with redundancy, searching for different queries and getting results that require normalization.
We need to rescore them with more semantic approach than simple index search.
""",
                high_level="Given a list of scored documents and a purpose for current search, provide a list of their updated scores.",
                low_level="""
You will be given full content of the input documents, as well as their metadata and current score.
Input score will always be in [0.0, 1.0] range.
You will use that knowledge to produce integer scores in 1-10 range.
You will treat the content as the most important. All the metadata, including previous score, should have much lower priority, but otherwise uniform amongst available details.
You will emit explicit ordering of the items, from the best to the worst. You will also provide updated score for each
item. 
You will include each and every input document ID in the ordering. Every ID will appear in the ordering exactly once. No ID will get duplicated or missed. This part of your response must be consistent with the order that could be derived from updated scores. 
You will emit an appropriate rescoring item for each input item. There will be exactly one rescoring item for each input item. No input item will get duplicated or missed. If we were to sort the list of rescorings by the updated score descending, the order should match the ordering emitted in the other part of your response. 
You will correlate the items and their ordering by their document ID. 
You will explain your choices, one at the time, referring to the input data and your inherent knowledge.
""",
                detailed="""
When choosing the scores for the documents you should use 1-10 scale (without 0, all integers). 
This output scale is different from the input one on purpose. DO NOT PRODUCE [0.0, 1.0] FLOAT SCORES.
ONLY produce integer scores in the 1-10 range.
You can use the following to get the gist of what does each value mean.

For example, if the purpose of the search is solving quantum physics problem, then the scores might be:
- a daily newspaper -> 1
- high-school physics book -> 3
- university-level quantum physics book -> 7
- scientific paper explicitly tackling the problem -> 10

Notice that 0 is NOT a valid score, which is why the daily newspaper is given score 1.

If not sure whether to choose a number or one bigger, explain the bigger choice and reevaluate. 
If you choose the lower value at that point, include the original explanation of the bigger choice, explanation of the switch and explanation of the chosen value as if it were chosen originally. 
"""
        ),
            input_type=RerankingInput,
            output_type=RerankingOutput,
        )
)

