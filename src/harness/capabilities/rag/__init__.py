"""可替换的 Retrieval-Augmented Generation 契约和管线。"""

from harness.capabilities.rag.contracts import (
    AccessScope,
    Citation,
    DocumentChunk,
    DocumentIndexState,
    EmbeddingProvider,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    SourceDocument,
    VectorStore,
)
from harness.capabilities.rag.evaluation import (
    EvaluationCase,
    EvaluationMetrics,
    evaluate_retrieval,
)
from harness.capabilities.rag.pipeline import RAGPipeline

__all__ = [
    "AccessScope",
    "Citation",
    "DocumentChunk",
    "DocumentIndexState",
    "EmbeddingProvider",
    "EvaluationCase",
    "EvaluationMetrics",
    "RAGPipeline",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "SourceDocument",
    "VectorStore",
    "evaluate_retrieval",
]
