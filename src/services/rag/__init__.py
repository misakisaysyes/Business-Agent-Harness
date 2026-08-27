"""RAG 基础设施适配器。"""

from services.rag.embeddings import EmbeddingDimensionError, FastEmbedProvider
from services.rag.factory import RAGComponents, create_rag_components
from services.rag.ingestion import IndexFailure, IndexingReport, IngestionService
from services.rag.splitter import DocumentSplitter, TextSplitterConfig
from services.rag.vector_store import (
    CollectionConfigurationError,
    EmbeddingRetriever,
    InMemoryVectorStore,
    LangChainPGVectorStore,
)

__all__ = [
    "CollectionConfigurationError",
    "DocumentSplitter",
    "EmbeddingDimensionError",
    "EmbeddingRetriever",
    "FastEmbedProvider",
    "IndexFailure",
    "IndexingReport",
    "InMemoryVectorStore",
    "IngestionService",
    "LangChainPGVectorStore",
    "RAGComponents",
    "TextSplitterConfig",
    "create_rag_components",
]
