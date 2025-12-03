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
                       indexing_volume: IndexingVolumeStrategy, #fixme shitty idea
                       cutoff: float | int = 0.8
                       ) -> list[ScoredDocument]:
        q = len(queries)
        assert q
        rewrite_count = rewrite_count or max(2, int(log2(q)))
        expansion_count = expansion_count or max(3, int(log2(q)))
        if isinstance(cutoff, int):
            assert cutoff <= 100
            cutoff = cutoff / 100.0
        total_index_results = indexing_volume(q, rewrite_count, expansion_count)
        #todo add purpose to rephraser
        rephrasings = await self._rephrase(rewrite_count, expansion_count, queries)
        all_index_queries = list(set(queries + rephrasings))
        search_results = await self.index.query(all_index_queries, total_index_results)
        scored_docs = [ ScoredDocument.from_search_result(x) for x in search_results ]
        grouped = self._group_for_dedup(scored_docs)
        deduplicated = []
        for group in grouped:
            assert group
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                best_choice = await self._deduplicate(purpose, group)
                deduplicated.append(best_choice)
        deduped_sorted = sorted(deduplicated, key=lambda x: x.score, reverse=True)
        reranked = await self._rerank(purpose, deduped_sorted)
        reranked_sorted = sorted(reranked, key=lambda x: x.score, reverse=True)
        normalized = self._softmax(reranked_sorted)
        results = self._cutoff(normalized, cutoff)
        return results

    async def _rephrase(self, rewrite_count: int, expansion_count: int, queries: list[str]) -> list[str]:
        return (await query_rephraser.run(
            to_simple_xml(
                RephrasingInput(
                    rewrite_count=rewrite_count,
                    expansion_count=expansion_count,
                    initial_queries=queries
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


    #todo add deduplication_instructions to guide agent on what to prioritize
    async def _deduplicate(self, purpose: str, group: list[ScoredDocument]) -> ScoredDocument:
        return (await deduplicator.run(to_simple_xml(DeduplicationInput(
            purpose=purpose,
            documents=IndexedScoredDocument.from_scored_documents(group)
        )))).output

    #todo ditto, reranking instructions
    async def _rerank(self, purpose: str, documents: list[ScoredDocument]) -> list[ScoredDocument]:
        return (await reranker.run(to_simple_xml(RerankingInput(
            purpose=purpose,
            documents=documents
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
        # s.set_attribute("cutoff.reached", summed)
        return out