import asyncio
from logging import getLogger
from typing import assert_never, Awaitable

from logfire import instrument
from pydantic_ai import FunctionToolset

from docassist.agents.rag.data import ScoredDocument
from docassist.agents.rag.tool import SearchKBTool
from docassist.index.document import Document
from docassist.index.protocols import DocumentIndex
from docassist.sampling.protocols import SamplingController
from docassist.structure.materialize.agent import materialization_aide
from docassist.structure.materialize.models import MaterializationState, VariableValuation
from docassist.structure.materialize.tasks import EvaluateVariableOutput, evaluate_variable
from docassist.structure.model import DocumentSpecification, DocumentDefinition, VariableValues, ChapterSpecification, \
    ChapterDefinition, ArticleDefinition, ArticleSpecification, BindingScope, Variant, Expansion

logger = getLogger(__name__)


def _shadow(current: VariableValues, new: VariableValues) -> VariableValues:
    out = dict(current)
    out.update(new)
    return out

#todo typing
def _flatten(i):
    out = []
    for x in i:
        out.extend(x)
    return out


class Materializer:
    def __init__(self, index: DocumentIndex, sampling: SamplingController):
        self.index: DocumentIndex = index
        self.sampling: SamplingController = sampling

    @instrument()
    async def _materialize(self, specification: DocumentSpecification,
                     state: MaterializationState
                     ) -> list[DocumentDefinition]:
        match specification:
            case None:
                return []
            case ChapterDefinition() | ArticleDefinition() as definition:
                return [definition]
            case ChapterSpecification() as c:
                return await self._materialize_chapter(c, state)
            case ArticleSpecification() as a:
                return self._materialize_article(a, state)
            case BindingScope() as scope:
                return await self._materialize_scope(scope, state)
            case Variant() as v:
                return await self._materialize_variant(v, state)
            case Expansion() as e:
                return await self._materialize_expansion(e, state)
            case _ as never:
                assert_never(never)

    @instrument()
    async def _materialize_chapter(self, c: ChapterSpecification,
                             state: MaterializationState) -> list[ChapterDefinition]:
        new_name = state.format_text_template(c.name)
        new_preamble = state.format_text_template(c.preamble_description)
        new_afterword = state.format_text_template(c.afterword_description)
        new_state = state.add_ancestor(new_name, new_preamble)

        return [
            ChapterDefinition(
                new_name,
                new_preamble,
                new_afterword,
                _flatten([
                    await self._materialize(spec, new_state)
                    for spec in c.content
                ])
            )
        ]

    @instrument()
    def _materialize_article(self, a: ArticleSpecification,
                             state: MaterializationState) -> list[ArticleDefinition]:
            return [
                ArticleDefinition(
                    state.format_text_template(a.name),
                    state.format_text_template(a.description)
                )
            ]

    def _tools(self, sideload: list[Document]) -> FunctionToolset:
        tool = SearchKBTool(self.index, self.sampling, )

        async def search_knowledge_base(*, purpose: str, queries: list[str],
                                        rewrite_count: int | None = None, expansion_count: int | None = None,
                                        additional_rephrasing_instructions: str | None = None,
                                        additional_deduplication_instructions: str | None = None,
                                        additional_reranking_instructions: str | None = None,
                                        cutoff: float | int = 0.8
                                        ) -> list[ScoredDocument]:
            # fixme copypasted
            """
            Run a multi-stage retrieval pipeline and return scored documents relevant to a given purpose.

            This pipeline is more than a simple vector search. It orchestrates several LLM-powered steps, each of which
            contributes to expanding, cleaning, and ranking the search space:

            **1. Search-area expansion**
               - An agent generates query rewrites and auxiliary queries to broaden the retrieval space.

            **2. Vector index retrieval**
               - Documents are retrieved in high volume from the vector store.
               - Quality is intentionally unfiltered at this stage to maximize recall.

            **3. Deduplication / grouping**
               - Retrieved documents are clustered:
                   • Original source files form their own groups and are always promoted.
                   • Derived documents (e.g., chunks) are grouped by type and subject.
               - Each group is sent to a specialized agent, which selects a representative and may adjust its score.

            **4. Reranking**
               - Group representatives are evaluated by another agent that assigns new scores based on alignment
                 with the overall search purpose.

            **5. Softmax normalization**
               - Documents are sorted by score and normalized so scores sum to 1.
               - Relative score ratios from the reranker are preserved.

            **6. Cutoff-based selection**
               - Documents are collected in descending score order until their cumulative normalized score
                 reaches the cutoff threshold.
               - Only the collected documents are returned.

            :param purpose: The high-level reason for performing the search. Provided to all agents in the
                pipeline for contextual alignment.
            :param queries: The initial search terms or phrases.
            :param rewrite_count: Number of rewrites per original query during expansion. Defaults to
                `max(2, log2(len(queries)))`.
            :param expansion_count: Number of auxiliary expansion queries. Defaults to the maximum of:
                `max(3, log2(len(queries)), rewrite_count)`.
            :param additional_rephrasing_instructions: Extra guidance for the expansion agent. Useful for applying
                constraints such as “use only the vocabulary present in the original queries” or other domain-specific rules.
            :param additional_deduplication_instructions: Extra guidance for the deduplication agent. Can express preferences
                such as “prefer smaller chunks” or “prefer full documents over chunk extracts”.
            :param additional_reranking_instructions: Extra guidance for the reranking agent. Can express relevance and
                priority rules, such as ranking complete documents above chunked ones when their content overlaps.
            :param cutoff: The score accumulation threshold used during final selection. Accepts either a float in the
                range [0.0, 1.0] or an integer in the range [0, 100]. Defaults to 0.8 (Pareto 80%).
            :return: The final set of documents, sorted by decreasing score and annotated with their normalized relevance
                values.
                    """
            return await tool.invoke(
                purpose=purpose, queries=queries,
                rewrite_count=rewrite_count, expansion_count=expansion_count,
                additional_rephrasing_instructions=additional_rephrasing_instructions,
                additional_deduplication_instructions=additional_deduplication_instructions,
                additional_reranking_instructions=additional_reranking_instructions,
                cutoff=cutoff
            )
        #todo add rephrase tool

        return FunctionToolset(
            max_retries=3, require_parameter_descriptions=True,
            tools=[search_knowledge_base]
        )

    @instrument
    async def _resolve_variable(self, name, desc, state: MaterializationState) -> VariableValuation: #todo return type
        task = evaluate_variable(name, desc)

        out = await materialization_aide.run(
            task, EvaluateVariableOutput,
            toolsets=[
                self._tools(state.fact_docs())
            ]
        )
        valuation = VariableValuation(value=out.variable_value, explanation=out.explanation)
        return valuation

    @instrument()
    async def _materialize_scope[T](self, s: BindingScope[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        new_state = state
        for var_name, var_desc in s.variables.items():
            resolved = await self._resolve_variable(var_name, var_desc, state)
            new_state = (new_state
                     .set_variable(var_name, resolved)
                     .add_fact(f"The value of variable `{var_name}` is '{resolved.value}", resolved.explanation)
                     .add_fact(f"The value of variable `{var_name}` as defined as '{var_desc}' is '{resolved.value}", resolved.explanation))

        return await self._materialize(s.content, new_state)

    @instrument()
    async def _materialize_variant[T](self, v: Variant[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        assert False
        # logger.info(f"Gathering data to answer {v.question}")
        # with self.controller.defer_until_success():
        #     response = self.model.feed(expand_the_answers(ExpandAnswersInput(
        #         expansions_count=5,
        #         question=v.question,
        #         answers=[IndexedAnswer(index=idx, answer=a.explanation) for idx, a in enumerate(v.answers)]
        #     ), self.llmio))
        #     expansions = self.llmio.lax_parse(response.content, ExpandAnswersOutput)
        #     queries = expansions.expanded
        #     assert queries
        # logger.info(f"Queries: {queries}")
        # docs = self.index.find(queries, n=5).grouped()
        # docs = self._rerank(closed_question(v.question, [a.explanation for a in v.answers]), docs)
        # logger.info("RAG result:")
        # for i, x in enumerate(docs):
        #     logger.info(f"DOCUMENT {i + 1}/{len(docs)}:")
        #     logger.info(x.entry.id + " //// " + str(x.score))
        #     logger.info(x.entry.txt)
        #     logger.info("-" * 80)
        # logger.info(f"Figuring out: {v.question}")
        # # fixme we're not using ancestors here
        # response = self.model.feed(
        #     choose_the_answer(
        #         ChooseTheAnswerInput(
        #             question=v.question,
        #             answers=[IndexedAnswer(index=i+1, answer=a.explanation) for i, a in enumerate(v.answers)],
        #             resources=DocumentChunk.from_items(
        #                 docs + [ #todo these should be added in reranking stage
        #                     ScoredItem(
        #                         entry=IndexedEntry.make(
        #                             "memory:processing_notes",
        #                             {},
        #                             self.llmio.dump(Facts(facts=state.facts))
        #                         ),
        #                         score=1.0
        #                     )
        #                 ] if state.facts else []
        #             )
        #         ),
        #         self.llmio
        #     )
        # )
        # choice = self.llmio.lax_parse(response.content, ChooseTheAnswerOutput)
        # idx = int(choice.chosen_index) - 1 # we give answers with 1-based index, we want 0-based one here
        # logger.info(f"LLM chose answer #{idx+1}: {v.answers[idx] if idx < len(v.answers) and idx > 0 else 'I dont know'}")
        # logger.info(f"The explanation of the choice was: <<{choice.explanation}>>")
        # assert idx < len(v.answers), "The LLM cannot reply with 'I dont know'"
        # chosen = v.answers[idx]
        # logger.info(f"Proceeding with {chosen}")
        # #todo add fact 'when presented with question ... we decided ...[ because ...]'
        # chosen_materialized = self._materialize(chosen.value, state)
        # return chosen_materialized

    @instrument()
    async def _materialize_expansion[T](self, exp: Expansion[T],
                                  state: MaterializationState) -> list[DocumentDefinition]:
        assert False
        # out = []
        # # fixme we're not using ancestors here
        # response = self.model.feed(
        #     expand_domain(
        #         input=ExpandOnDomainInput(
        #             domain_description=exp.domain_description,
        #             variables=exp.variables,
        #             resources=self.rag.retrieve(
        #                 index=self.index,
        #                 task=expansion(exp.domain_description),
        #                 queries=[exp.domain_description]
        #             )
        #         ),
        #         llmio=self.llmio
        #     )
        # )
        # result = self.llmio.lax_parse(response.content, ExpandOnDomainOutput).results
        # logger.info("Examples:")
        # for r in result:
        #     logger.info(pformat(r))
        #     new_state = state
        #     for k, v in r.values.items():
        #         new_state = (new_state
        #                      .set_variable(k, v)
        #                      .add_fact(f"The value of variable '{k}' defined as '{exp.variables[k]}' is '{v}'."))
        #     out.extend(self._materialize(exp.content_template, new_state))
        # return out

    @instrument()
    async def materialize_specification(self, specification: DocumentSpecification) -> list[DocumentDefinition]:
        return await self._materialize(specification, state=MaterializationState(ancestors=[], variable_values={}, facts=[]))