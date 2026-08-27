"""从校验配置创建共享 RAG 基础设施。"""

from dataclasses import dataclass

from harness.capabilities.rag import RAGPipeline
from services.config import RAGSettings
from services.rag.embeddings import FastEmbedProvider
from services.rag.vector_store import EmbeddingRetriever, LangChainPGVectorStore


@dataclass(frozen=True, slots=True)
class RAGComponents:
    embeddings: FastEmbedProvider
    store: LangChainPGVectorStore
    retriever: EmbeddingRetriever
    pipeline: RAGPipeline


def create_rag_components(settings: RAGSettings) -> RAGComponents:
    """创建一个可由所有用户 Runtime 共享、查询时再绑定 Scope 的 RAG Pipeline。"""

    if settings.database_url is None:
        raise ValueError("RAG database URL is required when RAG is enabled or indexing")
    embeddings = FastEmbedProvider(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    store = LangChainPGVectorStore(
        database_url=settings.database_url.get_secret_value(),
        collection_name=settings.collection_name,
        embeddings=embeddings,
    )
    retriever = EmbeddingRetriever(embeddings, store)
    return RAGComponents(
        embeddings=embeddings,
        store=store,
        retriever=retriever,
        pipeline=RAGPipeline(
            retriever,
            max_context_characters=settings.max_context_characters,
        ),
    )


__all__ = ["RAGComponents", "create_rag_components"]
