from docassist.agents.rag.data import RerankedItem, RerankingInput, RerankingOutput, ScoredDocument
from docassist.parametrized import Parametrized
from docassist.preindexing.perspectives import PERSPECTIVES, perspective
from docassist.structured_agent import StructuredAgent, DoerAgent, SolverAgent
from docassist.system_prompts import PromptingTask, Example

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
We already performed deduplication.
We need to rescore them with more semantic approach than simple index search.
""",
                high_level="Given a list of scored documents and a purpose for current search, provide updated scores "
                           "of all the input documents.",
                low_level="""
You will be given full content of the input documents, as well as their metadata and current score.
Input score will always be in [0.0, 1.0] range.
You will use that knowledge to produce integer scores in 1-10 range.
When coming up with updated scores, you will treat the content as the most important. All the metadata, including previous score, should have much lower priority, but otherwise uniform amongst available details.
You will refer to the input documents by their item index.
You will provide updated score for each item. 
You will emit an appropriate rescoring item for each input item. 
There will be exactly one rescoring item for each input item. No input item will get duplicated or omitted.  
You will explain your choices, one at the time, referring to the input data and your inherent knowledge.
""",

                detailed="""
When choosing the scores for the documents you will use 1-10 scale (without 0, all integers). 
This output scale is different from the input one on purpose. 
DO NOT PRODUCE [0.0, 1.0] FLOAT SCORES.
ONLY produce integer scores in the 1-10 range.

You can use the following to get the gist of what does each value mean.

For example, if the purpose of the search is solving quantum physics problem, then the scores might be:
- a daily newspaper -> 1
- high-school physics book -> 3
- university-level quantum physics book -> 7
- scientific paper explicitly tackling the problem -> 10

0 is NOT a valid score, which is why the daily newspaper is given score 1. 
DO NOT ever produce score of 0 or higher than 10. ONLY produce integer scores.

If not sure whether to choose a number or one bigger, pick the bigger value and explain your dillema.
"""
        ),
            # examples=[
            #     Example(
            #         input={
            #             "purpose"="find the crossword word for 'you see it in every movie happening in Paris",
            #             "documents"=[
            #                 ScoredDocument
            #             ]
            #         }
            #     )
            # ],
            input_type=RerankingInput,
            output_type=list[RerankedItem],
            # output_type=RerankingOutput,
        )
)

