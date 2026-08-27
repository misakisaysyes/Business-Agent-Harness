"""小型固定检索集的确定性 Recall/MRR/引用/隔离评测。"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from harness.capabilities.rag.citations import validate_citations
from harness.capabilities.rag.contracts import AccessScope, RetrievalQuery
from harness.capabilities.rag.pipeline import RAGPipeline


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    expected_sources: tuple[str, ...] = Field(min_length=1)


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cases: int = Field(ge=1)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    isolation_failures: int = Field(ge=0)


def evaluate_retrieval(
    pipeline: RAGPipeline,
    cases: Sequence[EvaluationCase],
    access_scope: AccessScope,
    *,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> EvaluationMetrics:
    """对固定 Query/来源标签计算首版客观检索指标。"""

    if not cases:
        raise ValueError("evaluation requires at least one case")
    recalls = reciprocal_ranks = valid_citations = isolation_failures = 0
    for case in cases:
        result = pipeline.search(
            RetrievalQuery(
                text=case.query,
                access_scope=access_scope,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        )
        sources = [hit.chunk.metadata.get("source") for hit in result.matches]
        expected = set(case.expected_sources)
        if expected.intersection(sources):
            recalls += 1
        first_rank = next(
            (index for index, source in enumerate(sources, start=1) if source in expected),
            None,
        )
        if first_rank is not None:
            reciprocal_ranks += 1 / first_rank
        try:
            validate_citations(result.citations, result.matches)
        except ValueError:
            pass
        else:
            valid_citations += 1
        for hit in result.matches:
            metadata = hit.chunk.metadata
            if metadata.get("scope") == "user" and metadata.get("user_id") != access_scope.user_id:
                isolation_failures += 1

    count = len(cases)
    return EvaluationMetrics(
        cases=count,
        recall_at_k=recalls / count,
        mrr=reciprocal_ranks / count,
        citation_validity=valid_citations / count,
        isolation_failures=isolation_failures,
    )


__all__ = ["EvaluationCase", "EvaluationMetrics", "evaluate_retrieval"]
