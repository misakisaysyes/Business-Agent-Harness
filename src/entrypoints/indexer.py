"""RAG 文档索引入口。

RAG document-indexing entry point.
"""

from pathlib import Path
from typing import Literal

from harness.logging import AgentLog, new_trace_id
from services.config import AppSettings
from services.rag import (
    DocumentSplitter,
    IndexingReport,
    IngestionService,
    TextSplitterConfig,
    create_rag_components,
)

log = AgentLog(__name__)


def index_documents(
    settings: AppSettings,
    source: str | Path,
    *,
    scope: Literal["public", "user"],
    user_id: str | None = None,
    rebuild: bool = False,
) -> IndexingReport:
    """从 CLI 参数和统一配置创建依赖并执行同步索引。"""

    with (
        log.bind(trace_id=new_trace_id()),
        log.operation("rag.indexer", scope=scope, rebuild=rebuild) as outcome,
    ):
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
        report = service.index_directory(
            source,
            scope=scope,
            user_id=user_id,
            rebuild=rebuild,
        )
        outcome["indexed_count"] = report.indexed
        outcome["skipped_count"] = report.skipped
        outcome["deleted_chunks"] = report.deleted_chunks
        outcome["failed_count"] = len(report.failed)
        if report.failed:
            outcome["status"] = "partial_failure"
        return report


__all__ = ["index_documents"]
