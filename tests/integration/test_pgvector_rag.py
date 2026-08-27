"""真实 PostgreSQL + pgvector 的持久化、Top-K 和用户隔离测试。"""

import os
import uuid
from collections.abc import Sequence

import pytest

from harness.capabilities.rag import AccessScope, DocumentChunk, DocumentIndexState
from services.rag.vector_store import LangChainPGVectorStore


class Fake512Embeddings:
    model_name = "fake-512-v1"
    dimension = 512

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        first = 1.0 if "refund" in text.casefold() else 0.0
        return (first, 1.0, *(0.0 for _ in range(510)))


@pytest.mark.rag_integration
def test_pgvector_persists_top_k_and_filters_user_scope() -> None:
    database_url = os.getenv("AGENT_RAG_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("AGENT_RAG_TEST_DATABASE_URL is not configured")
    collection = f"test_rag_{uuid.uuid4().hex}"
    store = LangChainPGVectorStore(database_url, collection, Fake512Embeddings())
    try:
        chunks = (
            _chunk("public", "public-doc", "public refund", "public", None),
            _chunk("alice", "alice-doc", "alice refund", "user", "alice"),
            _chunk("bob", "bob-doc", "bob refund", "user", "bob"),
        )
        for chunk in chunks:
            store.replace_document(
                _state(chunk),
                (chunk,),
                (Fake512Embeddings._vector(chunk.text),),
            )

        reopened = LangChainPGVectorStore(database_url, collection, Fake512Embeddings())
        hits = reopened.search(
            Fake512Embeddings._vector("refund"),
            AccessScope(user_id="alice"),
            10,
            {},
        )

        assert {hit.chunk.chunk_id for hit in hits} == {"public", "alice"}
        assert reopened.get_document_state("alice-doc") is not None
    finally:
        store.delete_collection()


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    scope: str,
    user_id: str | None,
) -> DocumentChunk:
    metadata = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "scope": scope,
        "source": f"{chunk_id}.md",
        "section": "Test",
    }
    if user_id is not None:
        metadata["user_id"] = user_id
    return DocumentChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
    )


def _state(chunk: DocumentChunk) -> DocumentIndexState:
    return DocumentIndexState(
        document_id=chunk.document_id,
        content_hash=chunk.chunk_id,
        embedding_model="fake-512-v1",
        embedding_dimension=512,
        splitter_version="v1",
        chunk_ids=(chunk.chunk_id,),
    )
