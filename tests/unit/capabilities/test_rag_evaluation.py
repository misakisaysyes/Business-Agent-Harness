"""固定检索集客观指标测试。"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from harness.capabilities.rag import (
    AccessScope,
    DocumentChunk,
    EvaluationCase,
    RAGPipeline,
    RetrievalHit,
    RetrievalQuery,
    evaluate_retrieval,
)


class ExpectedSourceRetriever:
    def __init__(self, cases: Sequence[EvaluationCase]) -> None:
        self.sources = {case.query: case.expected_sources[0] for case in cases}

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        source = self.sources[query.text]
        return (
            RetrievalHit(
                chunk=DocumentChunk(
                    document_id=source,
                    chunk_id=f"chunk-{source}",
                    text=f"Evidence from {source}",
                    metadata={"source": source, "section": "Test", "scope": "public"},
                ),
                score=1.0,
                rank=1,
            ),
        )


def test_fixed_evaluation_dataset_reports_recall_mrr_citations_and_isolation() -> None:
    fixture = Path("tests/fixtures/rag/evaluation.jsonl")
    cases = tuple(
        EvaluationCase.model_validate(cast(object, json.loads(line)))
        for line in fixture.read_text(encoding="utf-8").splitlines()
    )

    metrics = evaluate_retrieval(
        RAGPipeline(ExpectedSourceRetriever(cases)),
        cases,
        AccessScope(user_id="alice"),
    )

    assert metrics.cases == 5
    assert metrics.recall_at_k == 1
    assert metrics.mrr == 1
    assert metrics.citation_validity == 1
    assert metrics.isolation_failures == 0
