"""Retriever、Document 和检索结果契约。

Retriever, document, and retrieval-result contracts.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class RAGModel(BaseModel):
    """所有 RAG 数据契约共享的严格不可变配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AccessScope(RAGModel):
    """由可信 Runtime 注入的公共/用户数据访问范围。"""

    user_id: str | None = Field(default=None, min_length=1, max_length=64)
    include_public: bool = True

    @model_validator(mode="after")
    def require_one_scope(self) -> "AccessScope":
        if not self.include_public and self.user_id is None:
            raise ValueError("access scope must include public or one user")
        return self


class SourceDocument(RAGModel):
    """从受信目录加载、尚未切分的 UTF-8 文档。"""

    document_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=1_024)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DocumentChunk(RAGModel):
    """可独立向量化并能定位回原文的文本片段。"""

    document_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DocumentIndexState(RAGModel):
    """增量索引判断和旧 Chunk 清理所需的持久状态。"""

    document_id: str = Field(min_length=1, max_length=128)
    content_hash: str = Field(min_length=1, max_length=128)
    embedding_model: str = Field(min_length=1, max_length=512)
    embedding_dimension: int = Field(gt=0)
    splitter_version: str = Field(min_length=1, max_length=128)
    chunk_ids: tuple[str, ...] = ()


class RetrievalQuery(RAGModel):
    """一次带可信访问范围和有界参数的检索请求。"""

    text: str = Field(min_length=1, max_length=8_000)
    access_scope: AccessScope
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    filters: dict[str, JsonValue] = Field(default_factory=dict)


class RetrievalHit(RAGModel):
    """Retriever 返回的一个已授权候选片段。"""

    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class Citation(RAGModel):
    """一个确定性引用 ID 到真实 Chunk 的映射。"""

    id: str = Field(pattern=r"^S[1-9][0-9]*$")
    source: str = Field(min_length=1, max_length=1_024)
    section: str = Field(min_length=1, max_length=1_024)
    chunk_id: str = Field(min_length=1, max_length=128)


class RetrievalResult(RAGModel):
    """RAG Pipeline 输出给业务 Tool 的有界结果。"""

    matches: tuple[RetrievalHit, ...] = ()
    citations: tuple[Citation, ...] = ()
    context: str = ""
    message: str = Field(min_length=1)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """可替换的同步文本向量化协议。"""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """入库与检索共用的最小向量存储协议。"""

    def get_document_state(self, document_id: str) -> DocumentIndexState | None: ...

    def replace_document(
        self,
        state: DocumentIndexState,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        access_scope: AccessScope,
        top_k: int,
        filters: dict[str, JsonValue],
    ) -> Sequence[RetrievalHit]: ...


@runtime_checkable
class Retriever(Protocol):
    """RAG Pipeline 唯一允许调用的检索协议。"""

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]: ...


__all__ = [
    "AccessScope",
    "Citation",
    "DocumentChunk",
    "DocumentIndexState",
    "EmbeddingProvider",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "SourceDocument",
    "VectorStore",
]
