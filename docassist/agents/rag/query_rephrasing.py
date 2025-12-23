from pydantic_ai import Agent

from docassist.agents.rag.data import RephrasingOutput, RephrasingInput
from docassist.config import CONFIG
from docassist.llmio import object_from_user, object_from_llm
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask

#fixme should be a StructuredAgent
query_rephraser: Agent[None, RephrasingOutput]= Agent(
    name="query rephraser",
    model=CONFIG.model,
    output_type=RephrasingOutput,
    system_prompt=simple_xml_system_prompt(
        persona="RAG helper, specialised in rephrasing of queries",
        task=PromptingTask(
            context="you are at the beginning of the RAG pipeline",
            high_level="expand on given RAG queries to extend search area",
            low_level="given a list of RAG queries generate rewrites/rephrasings of queries and auxiliary queries",
            detailed=f"""You will be given a list of input queries and numbers that control how much additional content
you should generate. For each query you will generate `rewrite_count` queries that have similar meaning, but sound 
differently. Additionally, you will generate `expansion_count` queries that do not relate to one specific input query,
but are similar in meaning to all the input queries. You can think of rewrites as interpolation of search area, while
expansions are meant to simulate extrapolation.

# Rewriting rules

- Each rewrite must differ substantially in **structure**, not just word choice.
- You may vary:
  - active ↔ passive voice
  - clause splitting/merging
  - clause or phrase reordering
  - noun ↔ verb conversions
  - general word order
- Expansions may rephrase or expand the question context, but the outcome MUST be **standalone statements**.
- You will not include initial queries in the reponse. 

# Example

Input:

{
object_from_user(
    RephrasingInput(
        purpose="research into forest animals",
        initial_queries=[
            'Fox is faster than a rabbit.',
            'Rabbits are able to outrun foxes.'
        ],
        rewrite_count=2,
        expansion_count=3,
        additional_instructions="ignore the fact that fox and rabbit are common cartoon and child stories characters"
    )
)
}

Output:

{
object_from_llm(
    RephrasingOutput(
        rewrites=[
            ['A rabbit is slower when compared to a fox.', 'A fox can run at greater speed than a rabbit can.'],
            ['Compared with foxes, rabbits achieve higher speed.', 'A rabbit can surpass a fox in terms of quickness.']
        ],
        expansions=[
            'Speed of fox and rabbit.', 'Fox and rabbit in terms of speed.', "Comparison of forest animal speed"
        ]
    )
)
}
""",

        )
    )
)
