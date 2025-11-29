import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from docassist.config import CONFIG
from docassist.llmio import object_from_user, object_from_llm
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask


class RephrasingInput(BaseModel):
    rewrite_count: int
    expansion_count: int
    initial_queries: list[str]


class RephrasingOutput(BaseModel):
    rewrites: list[list[str]]
    expansions: list[str]

    def total(self) -> list[str]:
        def i():
            for r in self.rewrites:
                yield from r
            yield from self.expansions
        return list(i())

query_exploder = Agent(
    name="multi-query exploder",
    model=CONFIG.model,
    output_type=RephrasingOutput,
    system_prompt=simple_xml_system_prompt(
        persona="RAG helper",
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
        rewrite_count=2,
        expansion_count=3,
        initial_queries=[
            'Fox is faster than a rabbit.',
            'Rabbits are able to outrun foxes.'
        ]
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


# async def x():
#     r  = await query_exploder.run(
#             object_from_user(
#                 RephrasingInput(
#                     rewrite_count=3,
#                     expansion_count=2,
#                     initial_queries=["Size of the earth", "Diameter of the globe", "Length of equator"]
#                 )
#             )
#     )
#     print(type(r.output))
#     print(r.output)
#     print(r.output.total())
#
#     print(len(r.output.total()))
# asyncio.run(x())