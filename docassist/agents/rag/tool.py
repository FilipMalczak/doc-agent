from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from math import log2, exp
from typing import assert_never

from opentelemetry.trace import get_current_span
from pydantic_ai import ModelRetry

from docassist.agents.rag.data import ScoredDocument, DeduplicationInput, IndexedScoredDocument, RerankingInput, \
    RephrasingOutput, RerankingOutput
from docassist.agents.rag.deduplication import deduplicator
from docassist.agents.rag.query_rephrasing import query_rephraser, RephrasingInput
from docassist.agents.rag.reranking import reranker
from docassist.index.protocols import Document, DocumentIndex
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
    deduplicated: list[ScoredDocument] = field(default_factory=list)
    reranked: list[ScoredDocument] = field(default_factory=list)
    normalized: list[ScoredDocument] = field(default_factory=list)
    final: list[ScoredDocument] = field(default_factory=list)


#todo add spans over this
class SearchKBTool:
    def __init__(self, index: DocumentIndex, sampling: SamplingController, sideloaded_documents: list[Document] = []):
        self.index: DocumentIndex = index
        self.sideload = sideloaded_documents
        self.sampling = sampling

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
        rephrasings = await query_rephraser.run(
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
        s.set_attribute("groups_len", len(grouped))

        deduplicated = []
        for group in grouped:
            assert group
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                best_choice = await self._deduplicate(state.params.purpose, group, state.params.additional_deduplication_instructions)
                deduplicated.append(best_choice)
        s.set_attribute("deduplicated_len", len(deduplicated))

        state.deduplicated.extend(sorted(deduplicated, key=lambda x: x.score, reverse=True))

    def _group_for_dedup(self, chunks: list[ScoredDocument]) -> list[list[ScoredDocument]]:
        sources = list()
        chunkable_by_id: dict[str, Document] = dict()
        chunks_by_source_id: dict[str, list[Document]] = defaultdict(list)

        for chunk in chunks:
            match chunk.document.metadata.document_type:
                case "source_file":
                    assert chunk not in sources
                    sources.append(chunk)
                case "note" | "facts":
                    assert chunk.document.metadata.subject_path not in chunkable_by_id
                    chunkable_by_id[chunk.document.id] = chunk
                case "chunk":
                    chunks_by_source_id[chunk.document.metadata.chunk_source_document_id()].append(chunk)
                case _ as never: assert_never(never)

        def i():
            for s in sources:
                yield [s]
            visited_cids = set() # cid = chunk source id
            for cid in chain(chunkable_by_id.keys(), chunks_by_source_id.keys()):
                if cid not in visited_cids:
                    visited_cids.add(cid)
                    full = [ chunkable_by_id[cid] ] if cid in chunkable_by_id else []
                    yield full + chunks_by_source_id[cid]

        return list(i())


    async def _deduplicate(self, purpose: str, group: list[ScoredDocument], instructions: str) -> ScoredDocument:
        dedup_output = await deduplicator.run(DeduplicationInput(
                purpose=purpose,
                documents=IndexedScoredDocument.from_scored_documents(group),
                additional_instructions=instructions
            ))
        chosen = [ x for x in group if x.document.id == dedup_output.document_id ]
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
        reranked = await self._rerank(
            state.params.purpose,
            state.deduplicated,
            state.params.additional_reranking_instructions
        )
        state.reranked.extend(sorted(reranked, key=lambda x: x.score, reverse=True))

    async def _rerank(self, purpose: str, documents: list[ScoredDocument], instructions: str) -> list[ScoredDocument]:
        reranked: list[RerankingOutput] = await reranker.run(
            RerankingInput(
                purpose=purpose,
                documents=documents,
                additional_instructions=instructions
            )
        )
        docs_by_ids = {
            d.document.id: d for d in documents
        }
        reranks_by_id = {
            r.document_id: r for r in reranked
        }
        #todo these should be handled by StructuredAgent
        missing_input = set(docs_by_ids.keys()).difference(set(reranks_by_id.keys()))
        if missing_input:
            raise ModelRetry(f"Following document IDs were present in the input, but not the output: {', '.join(missing_input)}")
        hallucinated_output = set(reranks_by_id.keys()).difference(set(docs_by_ids.keys()))
        if hallucinated_output:
            raise ModelRetry(f"Following document IDs were present in the output, but not the input: {', '.join(hallucinated_output)}")

        return [
            doc.rescore(reranks_by_id[doc.document.id].new_score)
            for doc in documents
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

