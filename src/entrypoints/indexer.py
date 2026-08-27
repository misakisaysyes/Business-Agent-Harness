"""RAG 文档索引入口。

RAG document-indexing entry point.
"""

from pathlib import Path
from typing import Literal

from services.config import AppSettings
from services.rag import (
    DocumentSplitter,
    IndexingReport,
    IngestionService,
    TextSplitterConfig,
    create_rag_components,
)


def index_documents(
    settings: AppSettings,
    source: str | Path,
    *,
    scope: Literal["public", "user"],
    user_id: str | None = None,
    rebuild: bool = False,
) -> IndexingReport:
    """从 CLI 参数和统一配置创建依赖并执行同步索引。"""

    components = create_rag_components(settings.rag)
    splitter = DocumentSplitter(
        TextSplitterConfig(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
    )
    service = IngestionService(
        components.embeddings,
        components.store,
        splitter,
        knowledge_base_id=settings.rag.knowledge_base_id,
    )
    return service.index_directory(
        source,
        scope=scope,
        user_id=user_id,
        rebuild=rebuild,
    )


__all__ = ["index_documents"]
