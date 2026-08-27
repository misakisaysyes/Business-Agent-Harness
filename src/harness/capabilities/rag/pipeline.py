"""与具体技术实现解耦的 RAG 查询管线。

Framework-independent RAG query pipeline.
"""

from harness.capabilities.rag.citations import create_citations, render_context
from harness.capabilities.rag.contracts import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)


class RAGPipeline:
    """完成稳定排序、去重、阈值、预算和引用生成。"""

    def __init__(self, retriever: Retriever, max_context_characters: int = 12_000) -> None:
        if max_context_characters < 1:
            raise ValueError("max_context_characters must be positive")
        self.retriever = retriever
        self.max_context_characters = max_context_characters

    def search(self, query: RetrievalQuery) -> RetrievalResult:
        """执行一次有界检索；无结果时绝不生成引用。"""

        candidates = sorted(
            self.retriever.retrieve(query),
            key=lambda hit: (-hit.score, hit.rank, hit.chunk.chunk_id),
        )
        selected: list[RetrievalHit] = []
        seen_chunk_ids: set[str] = set()
        used_characters = 0

        for hit in candidates:
            if hit.chunk.chunk_id in seen_chunk_ids or hit.score < query.score_threshold:
                continue
            if len(selected) >= query.top_k:
                break
            chunk_characters = len(hit.chunk.text)
            if used_characters + chunk_characters > self.max_context_characters:
                continue
            selected.append(hit.model_copy(update={"rank": len(selected) + 1}))
            seen_chunk_ids.add(hit.chunk.chunk_id)
            used_characters += chunk_characters

        if not selected:
            return RetrievalResult(
                message="No authorized knowledge-base matches were found.",
            )

        hits = tuple(selected)
        citations = create_citations(hits)
        return RetrievalResult(
            matches=hits,
            citations=citations,
            context=render_context(hits, citations),
            message=f"Found {len(hits)} authorized knowledge-base match(es).",
        )


__all__ = ["RAGPipeline"]
