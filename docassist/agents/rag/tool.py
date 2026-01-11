import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from math import log2, exp
from typing import assert_never, Iterable

from opentelemetry.trace import get_current_span
from pydantic_ai import ModelRetry

from docassist.agents.rag.data import ScoredDocument, DeduplicationInput, IndexedScoredDocument, RerankingInput, \
    RephrasingOutput, RerankedItem, RerankingOutput
from docassist.agents.rag.deduplication import deduplicator
from docassist.agents.rag.query_rephrasing import query_rephraser, RephrasingInput
from docassist.agents.rag.reranking import reranker
from docassist.index.protocols import Document, DocumentIndex
from docassist.preindexing.perspectives import FINAL_DOCUMENTATION_PERSPECTIVE, FINAL_DOCUMENTATION_PERSPECTIVE_POINTER
from docassist.retries import phase, step
from docassist.sampling.protocols import SamplingController
from docassist.structured_agent import call_agent


@dataclass
class SearchParams:
    purpose: str
    queries: list[str]
    rewrite_count: int
    expansion_count: int
    additional_rephrasing_instructions: str
    additional_deduplication_instructions: str
    additional_reranking_instructions: str
    cutoff: float = 0.8

@dataclass
class SearchState:
    params: SearchParams
    rephrasings: list[str] = field(default_factory=list)
    scored_docs: list[ScoredDocument] = field(default_factory=list)
    grouped_for_dedup: list[list[ScoredDocument]] = field(default_factory=list)
    deduplicated: list[ScoredDocument] = field(default_factory=list)
    reranked: list[ScoredDocument] = field(default_factory=list)
    normalized: list[ScoredDocument] = field(default_factory=list)
    final: list[ScoredDocument] = field(default_factory=list)


def sliding_overlapping_window[T](data: list[T], step: int, max_len: int) -> Iterable[list[T]]:
    ks = 0 # k-times-s; we just add step per iteraiton instead of multiplying each time
    l = len(data)
    while ks < l:
        yield data[ks:ks+max_len]
        ks += step

def _ordered(sd: list[ScoredDocument]) -> list[ScoredDocument]:
    return sorted(sd, key=lambda x: x.score, reverse=True)

#todo add spans over this
class SearchKBTool:
    def __init__(self, index: DocumentIndex, sampling: SamplingController, sideloaded_documents: list[Document] = []):
        self.index: DocumentIndex = index
        self.sideload = sideloaded_documents
        self.sampling = sampling
        self._reranking_capacity: int = {
            'openai/gpt-oss-120b': 10,
            'deepseek/deepseek-v3.2': 10,
            'qwen/qwen3-32b': 8,
            'openai/gpt-oss-20b': 6
        }[reranker.parametrized_with(FINAL_DOCUMENTATION_PERSPECTIVE_POINTER).pydantic_agent.model.model_name]
        """
        Map the model names to reranking capacities here. Read the docstring for _rerank_stage(...) method.
        """

    def tool_name(self) -> str:
        return "search_knowledge_base"


    @property
    def __name__(self) -> str:
        return self.tool_name()

    async def invoke(self, purpose: str, queries: list[str], *,
                       rewrite_count: int | None = None, expansion_count: int | None = None,
                       additional_rephrasing_instructions: str | None = None,
                       additional_deduplication_instructions: str | None = None,
                       additional_reranking_instructions: str | None = None,
                       cutoff: float | int = 0.8
                       ) -> list[ScoredDocument]:
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
        state = SearchState(
            self._prepare_params(
                purpose,
                queries,
                rewrite_count,
                expansion_count,
                additional_rephrasing_instructions,
                additional_deduplication_instructions,
                additional_reranking_instructions,
                cutoff
            )
        )
        await self._rephrase_stage(state)
        await self._retrieve_stage(state)
        await self._deduplicate_stage(state)
        #sideload is added to deduplication results because sideloaded documents are meant to be out of index and unique
        # their score is Pareto 0.8 - we assume they might be useful, but we're not sure; reranking happens anyways
        state.deduplicated.extend([ScoredDocument(score=0.8, document=d) for d in self.sideload])
        await self._rerank_stage(state)
        await self._normalize_stage(state)
        self._choose_final_results(state)
        return state.final

    @step
    def _prepare_params(self, purpose: str, queries: list[str],
                       rewrite_count: int | None = None, expansion_count: int | None = None,
                       additional_rephrasing_instructions: str | None = None,
                       additional_deduplication_instructions: str | None = None,
                       additional_reranking_instructions: str | None = None,
                       cutoff: float | int = 0.8) -> SearchParams:
        s = get_current_span()
        q = len(queries)
        assert q
        rewrite_count = rewrite_count or max(2, int(log2(q)))
        expansion_count = expansion_count or max(3, int(log2(q)), rewrite_count)
        if isinstance(cutoff, int):
            assert cutoff <= 100
            cutoff = cutoff / 100.0
        assert cutoff >= 0
        # todo maybe each step should have its own span?
        # todo rewrite to set_attributes (plural)
        s.set_attribute("queries_len", q)
        s.set_attribute("effective_rewrite_count", rewrite_count)
        s.set_attribute("effective_expansion_count", expansion_count)
        s.set_attribute("effective_cutoff", cutoff)
        return SearchParams(
            purpose,
            queries,
            rewrite_count,
            expansion_count,
            additional_rephrasing_instructions or "",
            additional_deduplication_instructions or "",
            additional_reranking_instructions or "",
            cutoff
        )

    @phase()
    async def _rephrase_stage(self, state: SearchState):
        new_rephrasings = await self._rephrase(
            state.params.purpose,
            state.params.rewrite_count,
            state.params.expansion_count,
            state.params.queries,
            state.params.additional_rephrasing_instructions
        )
        state.rephrasings.extend(new_rephrasings)

    async def _rephrase(self, purpose: str, rewrite_count: int, expansion_count: int, queries: list[str], instructions) -> list[str]:
        #todo parametrize agents once and store them in self
        rephrasings = await query_rephraser.parametrized_with(FINAL_DOCUMENTATION_PERSPECTIVE_POINTER).run(
            RephrasingInput(
                purpose=purpose,
                rewrite_count=rewrite_count,
                expansion_count=expansion_count,
                initial_queries=queries,
                additional_instructions=instructions
            )
        )
        def i():
            for per_query in rephrasings.rewrites:
                yield from per_query
            yield from rephrasings.expansions
        return list(i())

    @phase()
    async def _retrieve_stage(self, state: SearchState):
        s = get_current_span()
        all_index_queries = list(set(state.params.queries + state.rephrasings))
        s.set_attribute("all_queries_len", len(all_index_queries))

        total_index_results = 2*len(all_index_queries)
        s.set_attribute("total_index_results", total_index_results)

        search_results = await self.index.query(all_index_queries, total_index_results)
        s.set_attribute("search_results_len", len(search_results))

        state.scored_docs.extend(ScoredDocument.from_search_result(x) for x in search_results)

    @phase()
    async def _deduplicate_stage(self, state: SearchState):
        s = get_current_span()
        grouped = self._group_for_dedup(state.scored_docs)
        state.grouped_for_dedup = grouped
        s.set_attribute("groups_len", len(grouped))

        deduplicated = []
        agent_backlog = []
        for group in grouped:
            assert group
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                agent_backlog.append(group)
        deduplicated.extend(
            await asyncio.gather(
                *(
                    self._deduplicate(
                        state.params.purpose,
                        group,
                        state.params.additional_deduplication_instructions
                    )
                    for group in agent_backlog
                )
            )
        )
        s.set_attribute("deduplicated_len", len(deduplicated))
        state.deduplicated.extend(_ordered(deduplicated))

    def _group_for_dedup(self, chunks: list[ScoredDocument]) -> list[list[ScoredDocument]]:
        sources = list()
        chunkable_by_id: dict[str, Document] = dict()
        chunks_by_source_id: dict[str, list[Document]] = defaultdict(list)

        deduplicated_by_id: dict[str, ScoredDocument] = {}
        for chunk in chunks:
            deduplicated_by_id[chunk.document.id] = chunk

        for chunk in deduplicated_by_id.values():
            match chunk.document.document_type: #todo these string constants should be taken from types
                case "source_file" | "transient":
                    assert chunk not in sources
                    sources.append(chunk)
                case "note" | "facts":
                    doc_id = chunk.document.id
                    assert doc_id not in chunkable_by_id
                    chunkable_by_id[doc_id] = chunk
                case "note_chapter" | "single_fact":
                    provenance = chunk.document.provenance
                    parent_id = provenance[-1].subject.id
                    chunks_by_source_id[parent_id].append(chunk)
                case _ as never: assert_never(never)

        def i():
            for s in sources:
                yield [s]
            visited_cids = set() # cid = chunk source id
            for cid in chain(chunkable_by_id.keys(), chunks_by_source_id.keys()):
                if cid not in visited_cids:
                    visited_cids.add(cid)
                    full = [ chunkable_by_id[cid] ] if cid in chunkable_by_id else []
                    # I can get literal duplicates (same ID, same metadata, etc) in input, because I'm searching across queries
                    # thus, I need to deduplicate; fixme it could be done beforehand
                    yield _ordered(full + chunks_by_source_id[cid])

        return list(i())


    async def _deduplicate(self, purpose: str, group: list[ScoredDocument], instructions: str) -> ScoredDocument:
        dedup_output = await deduplicator.parametrized_with(FINAL_DOCUMENTATION_PERSPECTIVE_POINTER).run(DeduplicationInput(
                purpose=purpose,
                documents=IndexedScoredDocument.from_scored_documents(group),
                additional_instructions=instructions
            ))
        chosen = [ x for x in group if x.document.id == dedup_output.document_id ]
        if not len(chosen) < 2:
            assert len(chosen) < 2
        if not chosen: #fixme modelretry restarts the whole tool; find the way to make reties per-stage
            raise ModelRetry(f"Returned document ID {dedup_output.document_id} does not match any input document!")
        if dedup_output.new_score < 0.0 or dedup_output.new_score > 1.0: #fixme this should be in model validator
            raise ModelRetry(f"Modified score must be in [0.0, 1.0] range! Returned score was {dedup_output.new_score} "
                             "instead!")
        rescored = chosen[0].rescore(dedup_output.new_score)
        return rescored

    @phase()
    async def _rerank_stage(self, state: SearchState):
        state.reranked = await self._rerank_dispatcher(state)

    async def _rerank_dispatcher(self, state: SearchState) -> list[ScoredDocument]:
        """
        Each model has limited capability of how many documents it can effectively rescore in one batch. For smaller
        models that's usually something like 8-10, with no model being able to realistically handle a hundred.

        Previous step might emit way more documents though, so we need to handle the over-the-capability situation.

        We have N input scored documents and a capability of rescoring M docs at once.
        We sort the input from the best to the worst.
        We apply sliding overlapping window with overlap O (0 < O < M).
        We apply reranker to each window.

        We need to reduce the scores from multiple window to a final result. We wanna account for documents from earlier
        windows being probably better than the trailing ones and we wanna account for the updated score from each window.

        Final formula is:

             score(doc) = (
                sum over windows containing doc:
                    reranker_score(doc, window) /
                    ( (window_index(window) + a) * (b + rank(doc, window))**2 )
            ) / (
                sum over windows containing doc: 1/(window_index(window) + a)
            )

        Since the number of windows for each element may vary (leading and trailing appearing less often than others)
        and since we're weighing by window index (to reflect "earlier = better"), we need to normalize after summing.

        `a` and `b` should amortize the window and document indexes/ranks, so that value of 1 doesn't overwhelm the rest
        of the sample. They should probably be around `a: int in [1, 5], b: int in [50, 100]`.
        """
        N = len(state.deduplicated)
        input_by_id = {
            d.document.id: d
            for d in state.deduplicated
        }
        M = self._reranking_capacity
        if N <= M:
            order: list[str, int] = await self._rerank(
                state.params.purpose,
                state.deduplicated,
                state.params.additional_reranking_instructions
            )
            return [
                input_by_id[id].rescore(new_score)
                for (id, new_score) in order
            ]

        overlap_factor = 0.75
        a = 5
        b = 50
        O = int(overlap_factor * M)
        S = M - O
        UpdatedScore = int  # 1-10
        WindowRank = int
        WindowIndex = int
        score_accumulator: dict[str, list[tuple[UpdatedScore, WindowRank, WindowIndex]]] = defaultdict(list)
        async def process_window(widx, window):
            window: list[ScoredDocument]
            order: list[str, int] = await self._rerank(
                state.params.purpose,
                window,
                state.params.additional_reranking_instructions
            )
            result = []
            for i, (id, new_score) in enumerate(order):
                result.append((new_score, i, widx))
            return {id: result}
        to_sum = await asyncio.gather(
            *(
                process_window(widx, window)
                for widx, window in enumerate(sliding_overlapping_window(state.deduplicated, S, M))
            )
        )
        for x in to_sum:
            score_accumulator.update(x)
        reduced_scores_by_id = {}
        for input_doc in state.deduplicated:
            id = input_doc.document.id
            upper = 0.0
            lower = 0.0
            for (score, rank, widx) in score_accumulator[id]:
                upper += ( score / ((widx + a)*((b + rank)**2)) )
                lower += ( 1 / (widx + a) )
            reduced_scores_by_id[id] = upper / lower
        return _ordered(
            [
                input_doc.rescore(
                    reduced_scores_by_id[input_doc.document.id]
                )
                for input_doc in state.deduplicated
            ]
        )


    async def _rerank(self, purpose: str, documents: list[ScoredDocument], instructions: str) -> list[tuple[str, int]]:
        """
        :return: list of (id, new_score) ordered by the agent
        """
        reranked: RerankingOutput = await reranker.parametrized_with(FINAL_DOCUMENTATION_PERSPECTIVE_POINTER).run(
            RerankingInput(
                purpose=purpose,
                documents=documents,
                additional_instructions=instructions
            )
        )
        docs_by_ids = {
            d.document.id: d for d in documents
        }
        rescores_by_id = {
            r.document_id: r for r in reranked.rescoring
        }
        missing_input = set(docs_by_ids.keys()).difference(set(rescores_by_id.keys()))
        if missing_input:
            #fixme wrong exception
            raise ModelRetry(f"Following document IDs were present in the input, but not the output: {', '.join(missing_input)}")
        hallucinated_output = set(rescores_by_id.keys()).difference(set(docs_by_ids.keys()))
        if hallucinated_output:
            raise ModelRetry(f"Following document IDs were present in the output, but not the input: {', '.join(hallucinated_output)}")
        ordering_by_score = sorted(rescores_by_id.keys(), key=lambda k: rescores_by_id[k].new_score, reverse=True)
        if not ordering_by_score == reranked.ordered_document_ids:
            raise ModelRetry(f"Explicit ordering of documents doesn't match score-induced ordering! (explicit: {reranked.ordered_document_ids}; by score: {ordering_by_score})")
        return [
            (id, rescores_by_id[id].new_score)
            for id in ordering_by_score
        ]

    @phase()
    async def _normalize_stage(self, state: SearchState):
        state.normalized.extend(self._softmax(state.reranked))

    def _softmax(self, data: list[ScoredDocument]) -> list[ScoredDocument]:
        exps = [exp(item.score) for item in data]
        sum_exps = sum(exps)
        return [
            ScoredDocument(document=item.document, score=exps[i] / sum_exps)
            for i, item in enumerate(data)
        ]

    @phase()
    def _choose_final_results(self, state: SearchState):
        s = get_current_span()
        state.final.extend(self._cutoff(state.normalized, state.params.cutoff))
        s.set_attribute("results_len", len(state.final))

    def _cutoff(self, data: list[ScoredDocument], cutoff: float) -> list[ScoredDocument]:
        out = []
        summed = 0.0
        for x in data:
            out.append(x)
            summed += x.score
            if summed >= cutoff:
                break
        return out

