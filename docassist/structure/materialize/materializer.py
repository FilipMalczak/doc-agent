from logging import getLogger
from typing import assert_never

from pydantic_ai import FunctionToolset, ModelRetry

from docassist.agents.rag.data import ScoredDocument
from docassist.agents.rag.tool import SearchKBTool
from docassist.index.document import Document
from docassist.index.protocols import DocumentIndex
from docassist.retries import phase, step
from docassist.sampling.protocols import SamplingController
from docassist.structure.materialize.agent import materialization_aide
from docassist.structure.materialize.expansion.graph import expand_domain
# from docassist.structure.materialize.agent import tooled_materialization_aide
from docassist.structure.materialize.models import MaterializationState, VariableValuation
from docassist.structure.materialize.tasks import EvaluateVariableOutput, evaluate_variable, choose_an_answer, \
    ChooseAnAnswerOutput
from docassist.structure.model import DocumentSpecification, DocumentDefinition, VariableValues, ChapterSpecification, \
    ChapterDefinition, ArticleDefinition, ArticleSpecification, BindingScope, Variant, Expansion, Answer

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

    @phase()
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

    @step
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

    @phase()
    def _materialize_article(self, a: ArticleSpecification,
                             state: MaterializationState) -> list[ArticleDefinition]:
            return [
                ArticleDefinition(
                    state.format_text_template(a.name),
                    state.format_text_template(a.description)
                )
            ]

    def _tools(self, sideload: list[Document]) -> FunctionToolset:
        tool = SearchKBTool(self.index, self.sampling, sideload)

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
            max_retries=3, require_parameter_descriptions=True, #todo retries count
            tools=[search_knowledge_base]
        )

    @phase()
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

    @phase()
    async def _materialize_scope[T](self, s: BindingScope[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        new_state = state
        for var_name, var_desc in s.variables.items():
            #fixme variables are resolved in isolation (all calls use old state); is this a good idea? maybe switch to resolving all at once too?
            resolved = await self._resolve_variable(var_name, var_desc, state)
            new_state = (new_state
                     .set_variable(var_name, resolved)
                     .add_fact(f"The value of variable `{var_name}` is '{resolved.value}", resolved.explanation)
                     .add_fact(f"The value of variable `{var_name}` as defined as '{var_desc}' is '{resolved.value}", resolved.explanation))

        return await self._materialize(s.content, new_state)

    @phase()
    async def _pick_an_answer[T](self, v: Variant[T],
                             state: MaterializationState) -> tuple[str, Answer[T]]:
        task, answer_ids = choose_an_answer(
            v.question,
            [a.explanation for a in v.answers]
        )
        # assert len(answer_ids) == len(v.answers)
        out = await materialization_aide.run(
            task, ChooseAnAnswerOutput,
            toolsets=[
                self._tools(state.fact_docs())
            ]
        )
        if out.correct_answer_id not in answer_ids:
            #fixme this is a wrong exception, it works only in tools and output functions; handle the retries on stage level
            raise ModelRetry(f"Answer ID `{out.correct_answer_id}` is not within available answer IDs: `{answer_ids}`")
        answer_idx = answer_ids.index(out.correct_answer_id)
        result = v.answers[answer_idx]
        return v.answers[answer_idx].explanation, result

    @phase()
    async def _materialize_variant[T](self, v: Variant[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        new_state = state
        text, picked = await self._pick_an_answer(v, state)
        new_state = new_state.add_fact(f"The answer to '{v.question}' is '{text}", picked.explanation)

        return await self._materialize(picked.value, new_state)


    @step
    async def _materialize_expansion[T](self, exp: Expansion[T],
                                  state: MaterializationState) -> list[DocumentDefinition]:
        response = await expand_domain(exp.domain_description, dict(exp.variables))
        assert False, f"Handle the response: {response}"

    @step
    async def materialize_specification(self, specification: DocumentSpecification) -> list[DocumentDefinition]:
        return await self._materialize(specification, state=MaterializationState(ancestors=[], variable_values={}, facts=[]))