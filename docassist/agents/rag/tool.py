from collections import defaultdict
from itertools import chain
from math import log2, exp
from typing import Callable, Self, assert_never

from pydantic import BaseModel

from docassist.agents.rag.data import ScoredDocument, DeduplicationInput, IndexedScoredDocument, RerankingInput
from docassist.agents.rag.deduplication import deduplicator
from docassist.agents.rag.query_rephrasing import query_rephraser, RephrasingInput
from docassist.agents.rag.reranking import reranker
from docassist.index.protocols import Document, DocumentIndex, SearchResult
from docassist.simple_xml import to_simple_xml


QueryCount, RewriteCount, ExpansionCount, ResultsCount = int, int, int, int
IndexingVolumeStrategy = Callable[[QueryCount, RewriteCount, ExpansionCount], ResultsCount]

#todo add spans over this
class SearchIndexTool:
    def __init__(self, index: DocumentIndex):
        self.index: DocumentIndex = index

    async def __call__(self, purpose: str, queries: list[str], *,
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
        q = len(queries)
        assert q
        rewrite_count = rewrite_count or max(2, int(log2(q)))
        expansion_count = expansion_count or max(3, int(log2(q)), rewrite_count)
        if isinstance(cutoff, int):
            assert cutoff <= 100
            cutoff = cutoff / 100.0
        assert cutoff >= 0
        #todo add purpose to rephraser
        rephrasings = await self._rephrase(purpose, rewrite_count, expansion_count, queries, additional_rephrasing_instructions or "")
        all_index_queries = list(set(queries + rephrasings))
        total_index_results = 2*len(all_index_queries)
        search_results = await self.index.query(all_index_queries, total_index_results)
        scored_docs = [ ScoredDocument.from_search_result(x) for x in search_results ]
        grouped = self._group_for_dedup(scored_docs)
        deduplicated = []
        for group in grouped:
            assert group
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                best_choice = await self._deduplicate(purpose, group, additional_deduplication_instructions or "")
                deduplicated.append(best_choice)
        deduped_sorted = sorted(deduplicated, key=lambda x: x.score, reverse=True)
        reranked = await self._rerank(purpose, deduped_sorted, additional_reranking_instructions or "")
        reranked_sorted = sorted(reranked, key=lambda x: x.score, reverse=True)
        normalized = self._softmax(reranked_sorted)
        results = self._cutoff(normalized, cutoff)
        return results

    async def _rephrase(self, purpose: str, rewrite_count: int, expansion_count: int, queries: list[str], instructions) -> list[str]:
        return (await query_rephraser.run(
            to_simple_xml(
                RephrasingInput(
                    purpose=purpose,
                    rewrite_count=rewrite_count,
                    expansion_count=expansion_count,
                    initial_queries=queries,
                    additional_instructions=instructions
                )
            )
        )).output

    def _group_for_dedup(self, chunks: list[ScoredDocument]) -> list[list[ScoredDocument]]:
        sources = list()
        chunkable_by_id: dict[str, Document] = dict()
        chunks_by_source_id: dict[str, set[Document]] = defaultdict(set)

        for chunk in chunks:
            match chunk.document.metadata.document_type:
                case "source_file":
                    assert chunk not in sources
                    sources.append(chunk)
                case "note" | "facts":
                    assert chunk.document.metadata.subject_path not in chunkable_by_id
                    chunkable_by_id[chunk.document.id] = chunk
                case "chunk":
                    # ooh, this is crap
                    chunks_by_source_id[chunk.document.metadata.chunk_source_document_id()].add(chunk)
                case _ as never: assert_never(never)

        def i():
            for s in sources:
                yield [s]
            visited_cids = set() # cid = chunk source id
            for cid in chain(chunkable_by_id.keys(), chunks_by_source_id.keys()):
                if cid not in visited_cids:
                    visited_cids.add(cid)
                    yield [ chunkable_by_id[cid] ] + list(chunks_by_source_id[cid])

        return list(i())


    async def _deduplicate(self, purpose: str, group: list[ScoredDocument], instructions: str) -> ScoredDocument:
        return (await deduplicator.run(to_simple_xml(DeduplicationInput(
            purpose=purpose,
            documents=IndexedScoredDocument.from_scored_documents(group),
            additional_instructions=instructions
        )))).output

    async def _rerank(self, purpose: str, documents: list[ScoredDocument], instructions: str) -> list[ScoredDocument]:
        return (await reranker.run(to_simple_xml(RerankingInput(
            purpose=purpose,
            documents=documents,
            additional_instructions=instructions
        )))).output

    def _softmax(self, data: list[ScoredDocument]) -> list[ScoredDocument]:
        exps = [exp(item.score) for item in data]
        sum_exps = sum(exps)
        return [
            ScoredDocument(document=item.document, score=exps[i] / sum_exps)
            for i, item in enumerate(data)
        ]

    def _cutoff(self, data: list[ScoredDocument], cutoff: float) -> list[ScoredDocument]:
        out = []
        summed = 0.0
        for x in data:
            out.append(x)
            summed += x.score
            if summed >= cutoff:
                break
        return out