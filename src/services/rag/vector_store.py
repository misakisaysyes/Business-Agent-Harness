"""Vector Store、Retriever 和 pgvector 适配器。

Vector-store, retriever, and pgvector adapters.
"""

import json
import math
from collections.abc import Sequence
from threading import RLock
from typing import Any

from pydantic import JsonValue

from harness.capabilities.rag import (
    AccessScope,
    DocumentChunk,
    DocumentIndexState,
    EmbeddingProvider,
    RetrievalHit,
    RetrievalQuery,
)
from harness.logging import AgentLog

log = AgentLog(__name__)


class CollectionConfigurationError(RuntimeError):
    """Collection 中存在不兼容的模型或向量维度。"""


def _metadata_matches(metadata: dict[str, JsonValue], filters: dict[str, JsonValue]) -> bool:
    for key, expected in filters.items():
        actual = metadata.get(key)
        if key == "tags" and isinstance(expected, list):
            if not isinstance(actual, list) or not set(expected).issubset(actual):
                return False
        elif actual != expected:
            return False
    return True


def _scope_allows(metadata: dict[str, JsonValue], scope: AccessScope) -> bool:
    item_scope = metadata.get("scope")
    if item_scope == "public":
        return scope.include_public
    return (
        item_scope == "user"
        and scope.user_id is not None
        and metadata.get("user_id") == scope.user_id
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return max(0.0, min(1.0, dot_product / (left_norm * right_norm)))


class InMemoryVectorStore:
    """默认测试使用的线程安全内存向量库。"""

    def __init__(self) -> None:
        self._records: dict[str, tuple[DocumentChunk, tuple[float, ...]]] = {}
        self._states: dict[str, DocumentIndexState] = {}
        self._configuration: tuple[str, int] | None = None
        self._lock = RLock()

    def get_document_state(self, document_id: str) -> DocumentIndexState | None:
        with self._lock:
            return self._states.get(document_id)

    def replace_document(
        self,
        state: DocumentIndexState,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if tuple(chunk.chunk_id for chunk in chunks) != state.chunk_ids:
            raise ValueError("document state chunk IDs do not match chunks")
        if any(len(vector) != state.embedding_dimension for vector in vectors):
            raise ValueError("vector dimension does not match document state")
        with self._lock:
            configuration = (state.embedding_model, state.embedding_dimension)
            if self._configuration is not None and self._configuration != configuration:
                raise CollectionConfigurationError(
                    "cannot mix embedding models or dimensions in one collection"
                )
            previous = self._states.get(state.document_id)
            for chunk_id in previous.chunk_ids if previous is not None else ():
                self._records.pop(chunk_id, None)
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._records[chunk.chunk_id] = (chunk, tuple(float(value) for value in vector))
            self._states[state.document_id] = state
            self._configuration = configuration

    def search(
        self,
        vector: Sequence[float],
        access_scope: AccessScope,
        top_k: int,
        filters: dict[str, JsonValue],
    ) -> tuple[RetrievalHit, ...]:
        with self._lock:
            candidates = [
                (chunk, _cosine_similarity(vector, stored_vector))
                for chunk, stored_vector in self._records.values()
                if _scope_allows(chunk.metadata, access_scope)
                and _metadata_matches(chunk.metadata, filters)
            ]
        candidates.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return tuple(
            RetrievalHit(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(candidates[:top_k], start=1)
        )


class EmbeddingRetriever:
    """通过 EmbeddingProvider 和 VectorStore 实现通用 Retriever。"""

    def __init__(self, embeddings: EmbeddingProvider, store: Any) -> None:
        self.embeddings = embeddings
        self.store = store

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        with log.operation(
            "rag.embedding.query",
            embedding_model=self.embeddings.model_name,
            embedding_dimension=self.embeddings.dimension,
            query_characters=len(query.text),
        ) as embedding_outcome:
            vector = self.embeddings.embed_query(query.text)
            embedding_outcome["vector_count"] = 1
        with log.operation(
            "rag.vector_search",
            top_k=query.top_k,
            filter_count=len(query.filters),
            include_public=query.access_scope.include_public,
            has_user_scope=query.access_scope.user_id is not None,
        ) as search_outcome:
            hits = self.store.search(
                vector,
                query.access_scope,
                query.top_k,
                query.filters,
            )
            search_outcome["retrieved_count"] = len(hits)
            return hits


class _LangChainEmbeddingsAdapter:
    """把项目 EmbeddingProvider 暴露为 LangChain Embeddings 接口。"""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.provider.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self.provider.embed_query(text))


class LangChainPGVectorStore:
    """基于 langchain-postgres 的 pgvector Collection 与增量清单适配器。"""

    _MANIFEST_TABLE = "agent_rag_document_manifest"

    def __init__(
        self,
        database_url: str,
        collection_name: str,
        embeddings: EmbeddingProvider,
    ) -> None:
        if not database_url:
            raise ValueError("RAG database URL is required")
        from langchain_postgres import PGVector
        from sqlalchemy import create_engine, text

        self.collection_name = collection_name
        self.embeddings = embeddings
        self._text = text
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._store: Any = PGVector(
            embeddings=_LangChainEmbeddingsAdapter(embeddings),  # type: ignore[arg-type]
            connection=database_url,
            embedding_length=embeddings.dimension,
            collection_name=collection_name,
            collection_metadata={
                "embedding_model": embeddings.model_name,
                "embedding_dimension": embeddings.dimension,
            },
            use_jsonb=True,
            create_extension=True,
        )
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._MANIFEST_TABLE} (
                        collection_name TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dimension INTEGER NOT NULL,
                        splitter_version TEXT NOT NULL,
                        chunk_ids JSONB NOT NULL,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (collection_name, document_id)
                    )
                    """
                )
            )

    def get_document_state(self, document_id: str) -> DocumentIndexState | None:
        statement = self._text(
            f"""
            SELECT document_id, content_hash, embedding_model, embedding_dimension,
                   splitter_version, chunk_ids
            FROM {self._MANIFEST_TABLE}
            WHERE collection_name = :collection_name AND document_id = :document_id
            """
        )
        with self._engine.connect() as connection:
            row = connection.execute(
                statement,
                {"collection_name": self.collection_name, "document_id": document_id},
            ).mappings().first()
        if row is None:
            return None
        chunk_ids = row["chunk_ids"]
        if isinstance(chunk_ids, str):
            chunk_ids = json.loads(chunk_ids)
        return DocumentIndexState(
            document_id=row["document_id"],
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            splitter_version=row["splitter_version"],
            chunk_ids=tuple(chunk_ids),
        )

    def replace_document(
        self,
        state: DocumentIndexState,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        with log.operation(
            "rag.vector_store.replace_document",
            collection_name=self.collection_name,
            document_id=state.document_id,
            chunk_count=len(chunks),
            vector_count=len(vectors),
            embedding_model=state.embedding_model,
            embedding_dimension=state.embedding_dimension,
        ) as outcome:
            chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
            if len(chunks) != len(vectors) or chunk_ids != state.chunk_ids:
                raise ValueError("chunks, vectors, and document state are inconsistent")
            if any(len(vector) != state.embedding_dimension for vector in vectors):
                raise ValueError("vector dimension does not match document state")
            self._assert_collection_compatible(state)
            previous = self.get_document_state(state.document_id)
            self._store.add_embeddings(
                texts=[chunk.text for chunk in chunks],
                embeddings=[list(vector) for vector in vectors],
                metadatas=[dict(chunk.metadata) for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
            )
            old_ids = set(previous.chunk_ids if previous is not None else ()) - set(
                state.chunk_ids
            )
            if old_ids:
                self._store.delete(ids=sorted(old_ids), collection_only=True)
            statement = self._text(
                f"""
                INSERT INTO {self._MANIFEST_TABLE}
                    (collection_name, document_id, content_hash, embedding_model,
                     embedding_dimension, splitter_version, chunk_ids, indexed_at)
                VALUES
                    (:collection_name, :document_id, :content_hash, :embedding_model,
                     :embedding_dimension, :splitter_version, CAST(:chunk_ids AS JSONB), NOW())
                ON CONFLICT (collection_name, document_id) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimension = EXCLUDED.embedding_dimension,
                    splitter_version = EXCLUDED.splitter_version,
                    chunk_ids = EXCLUDED.chunk_ids,
                    indexed_at = NOW()
                """
            )
            with self._engine.begin() as connection:
                connection.execute(
                    statement,
                    {
                        "collection_name": self.collection_name,
                        "document_id": state.document_id,
                        "content_hash": state.content_hash,
                        "embedding_model": state.embedding_model,
                        "embedding_dimension": state.embedding_dimension,
                        "splitter_version": state.splitter_version,
                        "chunk_ids": json.dumps(state.chunk_ids),
                    },
                )
            outcome["deleted_chunks"] = len(old_ids)

    def search(
        self,
        vector: Sequence[float],
        access_scope: AccessScope,
        top_k: int,
        filters: dict[str, JsonValue],
    ) -> tuple[RetrievalHit, ...]:
        with log.operation(
            "rag.pgvector.search",
            collection_name=self.collection_name,
            top_k=top_k,
            filter_count=len(filters),
            include_public=access_scope.include_public,
            has_user_scope=access_scope.user_id is not None,
        ) as outcome:
            fetch_k = min(max(top_k * 4, top_k), 200)
            rows: list[tuple[Any, float]] = []
            public_count = 0
            user_count = 0
            if access_scope.include_public:
                public_rows = self._store.similarity_search_with_score_by_vector(
                    list(vector), k=fetch_k, filter={"scope": "public"}
                )
                public_count = len(public_rows)
                rows.extend(public_rows)
            if access_scope.user_id is not None:
                user_rows = self._store.similarity_search_with_score_by_vector(
                    list(vector),
                    k=fetch_k,
                    filter={"scope": "user", "user_id": access_scope.user_id},
                )
                user_count = len(user_rows)
                rows.extend(user_rows)

            candidates: list[tuple[DocumentChunk, float]] = []
            for document, distance in rows:
                metadata = dict(document.metadata)
                if not _metadata_matches(metadata, filters):
                    continue
                chunk_id = metadata.get("chunk_id")
                document_id = metadata.get("document_id")
                if not isinstance(chunk_id, str) or not isinstance(document_id, str):
                    continue
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=document.page_content,
                    metadata=metadata,
                )
                score = max(0.0, min(1.0, 1.0 - float(distance)))
                candidates.append((chunk, score))
            candidates.sort(key=lambda item: (-item[1], item[0].chunk_id))
            hits = tuple(
                RetrievalHit(chunk=chunk, score=score, rank=rank)
                for rank, (chunk, score) in enumerate(candidates[:top_k], start=1)
            )
            outcome["public_candidate_count"] = public_count
            outcome["user_candidate_count"] = user_count
            outcome["candidate_count"] = len(candidates)
            outcome["selected_count"] = len(hits)
            return hits

    def _assert_collection_compatible(self, state: DocumentIndexState) -> None:
        statement = self._text(
            f"""
            SELECT embedding_model, embedding_dimension
            FROM {self._MANIFEST_TABLE}
            WHERE collection_name = :collection_name
            LIMIT 1
            """
        )
        with self._engine.connect() as connection:
            row = connection.execute(
                statement, {"collection_name": self.collection_name}
            ).mappings().first()
        if row is not None and (
            row["embedding_model"] != state.embedding_model
            or row["embedding_dimension"] != state.embedding_dimension
        ):
            raise CollectionConfigurationError(
                "collection already uses a different embedding model or dimension"
            )

    def delete_collection(self) -> None:
        """删除当前 Collection 及其增量清单，主要用于隔离集成测试。"""

        self._store.delete_collection()
        statement = self._text(
            f"DELETE FROM {self._MANIFEST_TABLE} WHERE collection_name = :collection_name"
        )
        with self._engine.begin() as connection:
            connection.execute(statement, {"collection_name": self.collection_name})


__all__ = [
    "CollectionConfigurationError",
    "EmbeddingRetriever",
    "InMemoryVectorStore",
    "LangChainPGVectorStore",
]
