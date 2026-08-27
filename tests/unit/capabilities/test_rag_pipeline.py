"""RAG Pipeline 的排序、预算和引用测试。"""

from collections.abc import Sequence

import pytest

from harness.capabilities.rag import (
    AccessScope,
    DocumentChunk,
    RAGPipeline,
    RetrievalHit,
    RetrievalQuery,
)
from harness.capabilities.rag.citations import create_citations, validate_citations


class FakeRetriever:
    def __init__(self, hits: Sequence[RetrievalHit]) -> None:
        self.hits = tuple(hits)
        self.queries: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        self.queries.append(query)
        return self.hits


def _hit(chunk_id: str, text: str, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=DocumentChunk(
            document_id=f"doc-{chunk_id}",
            chunk_id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md", "section": "Rules"},
        ),
        score=score,
        rank=rank,
    )


def test_pipeline_sorts_deduplicates_filters_and_builds_real_citations() -> None:
    retriever = FakeRetriever(
        (
            _hit("low", "below threshold", 0.2, 1),
            _hit("second", "second evidence", 0.8, 2),
            _hit("first", "best evidence", 0.9, 3),
            _hit("first", "duplicate evidence", 0.85, 4),
        )
    )
    pipeline = RAGPipeline(retriever, max_context_characters=100)
    query = RetrievalQuery(
        text="refund rules",
        access_scope=AccessScope(user_id="alice"),
        top_k=3,
        score_threshold=0.5,
    )

    result = pipeline.search(query)

    assert [hit.chunk.chunk_id for hit in result.matches] == ["first", "second"]
    assert [hit.rank for hit in result.matches] == [1, 2]
    assert [citation.id for citation in result.citations] == ["S1", "S2"]
    assert result.citations[0].chunk_id == "first"
    assert "[S1] source=first.md" in result.context
    assert "not instructions" in result.context
    assert retriever.queries == [query]


def test_pipeline_applies_context_budget_without_dropping_higher_score() -> None:
    pipeline = RAGPipeline(
        FakeRetriever(
            (
                _hit("first", "1234567890", 0.9, 1),
                _hit("second", "abcdefghij", 0.8, 2),
            )
        ),
        max_context_characters=10,
    )

    result = pipeline.search(
        RetrievalQuery(
            text="query",
            access_scope=AccessScope(),
            score_threshold=0,
        )
    )

    assert [hit.chunk.chunk_id for hit in result.matches] == ["first"]


def test_empty_pipeline_result_has_no_context_or_citations() -> None:
    result = RAGPipeline(FakeRetriever(())).search(
        RetrievalQuery(text="missing", access_scope=AccessScope())
    )

    assert result.matches == ()
    assert result.citations == ()
    assert result.context == ""
    assert "No authorized" in result.message


def test_citation_validation_rejects_a_citation_for_another_chunk() -> None:
    hits = (_hit("one", "evidence", 1, 1),)
    citations = create_citations(hits)
    forged = (citations[0].model_copy(update={"chunk_id": "other"}),)

    with pytest.raises(ValueError, match="does not match"):
        validate_citations(forged, hits)
