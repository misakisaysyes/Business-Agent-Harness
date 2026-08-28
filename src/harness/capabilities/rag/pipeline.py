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
from harness.logging import AgentLog

log = AgentLog(__name__)


class RAGPipeline:
    """完成稳定排序、去重、阈值、预算和引用生成。"""

    def __init__(self, retriever: Retriever, max_context_characters: int = 12_000) -> None:
        if max_context_characters < 1:
            raise ValueError("max_context_characters must be positive")
        self.retriever = retriever
        self.max_context_characters = max_context_characters

    def search(self, query: RetrievalQuery) -> RetrievalResult:
        """执行一次有界检索；无结果时绝不生成引用。"""

        common_fields = {
            "query_characters": len(query.text),
            "top_k": query.top_k,
            "score_threshold": query.score_threshold,
            "max_context_characters": self.max_context_characters,
            "filter_count": len(query.filters),
            "include_public": query.access_scope.include_public,
            "has_user_scope": query.access_scope.user_id is not None,
        }
        with log.operation("rag.pipeline", **common_fields) as pipeline_outcome:
            with log.operation("rag.retrieve", **common_fields) as retrieval_outcome:
                retrieved = tuple(self.retriever.retrieve(query))
                retrieval_outcome["retrieved_count"] = len(retrieved)

            candidates = sorted(
                retrieved,
                key=lambda hit: (-hit.score, hit.rank, hit.chunk.chunk_id),
            )
            selected: list[RetrievalHit] = []
            seen_chunk_ids: set[str] = set()
            used_characters = 0
            duplicate_count = 0
            threshold_dropped_count = 0
            remaining_after_top_k_count = 0
            budget_dropped_count = 0

            for hit in candidates:
                if hit.chunk.chunk_id in seen_chunk_ids:
                    duplicate_count += 1
                    continue
                if hit.score < query.score_threshold:
                    threshold_dropped_count += 1
                    continue
                if len(selected) >= query.top_k:
                    remaining_after_top_k_count += 1
                    continue
                chunk_characters = len(hit.chunk.text)
                if used_characters + chunk_characters > self.max_context_characters:
                    budget_dropped_count += 1
                    continue
                selected.append(hit.model_copy(update={"rank": len(selected) + 1}))
                seen_chunk_ids.add(hit.chunk.chunk_id)
                used_characters += chunk_characters

            pipeline_outcome.update(
                {
                    "candidate_count": len(candidates),
                    "duplicate_count": duplicate_count,
                    "threshold_dropped_count": threshold_dropped_count,
                    "remaining_after_top_k_count": remaining_after_top_k_count,
                    "budget_dropped_count": budget_dropped_count,
                    "selected_count": len(selected),
                }
            )

            if not selected:
                pipeline_outcome["status"] = "no_matches"
                return RetrievalResult(
                    message="No authorized knowledge-base matches were found.",
                )

            hits = tuple(selected)
            citations = create_citations(hits)
            context = render_context(hits, citations)
            pipeline_outcome["context_characters"] = len(context)
            return RetrievalResult(
                matches=hits,
                citations=citations,
                context=context,
                message=f"Found {len(hits)} authorized knowledge-base match(es).",
            )


__all__ = ["RAGPipeline"]
