"""Embedding Provider 适配器。

Embedding-provider adapters.
"""

from collections.abc import Sequence
from threading import Lock
from typing import Any


class EmbeddingDimensionError(ValueError):
    """模型实际维度与 Collection 配置不一致。"""


class FastEmbedProvider:
    """延迟加载 FastEmbed，避免 RAG 关闭时导入模型运行时。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dimension: int = 512,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("embedding model name must not be empty")
        if dimension < 1:
            raise ValueError("embedding dimension must be positive")
        self._model_name = model_name
        self._dimension = dimension
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        vectors = self._model_instance().passage_embed(list(texts))
        return tuple(self._validated_vector(vector) for vector in vectors)

    def embed_query(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("embedding query must not be empty")
        vectors = self._model_instance().query_embed([text])
        try:
            vector = next(iter(vectors))
        except StopIteration as error:
            raise RuntimeError("embedding model returned no query vector") from error
        return self._validated_vector(vector)

    def _model_instance(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                from fastembed import TextEmbedding

                kwargs: dict[str, Any] = {"model_name": self.model_name}
                if self._cache_dir is not None:
                    kwargs["cache_dir"] = self._cache_dir
                if self._threads is not None:
                    kwargs["threads"] = self._threads
                self._model = TextEmbedding(**kwargs)
        return self._model

    def _validated_vector(self, vector: Any) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if len(values) != self.dimension:
            raise EmbeddingDimensionError(
                f"embedding dimension mismatch: expected {self.dimension}, got {len(values)}"
            )
        return values


__all__ = ["EmbeddingDimensionError", "FastEmbedProvider"]
