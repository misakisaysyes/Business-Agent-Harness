"""按权限枚举和统计知识库文档的业务 Tool。"""

import asyncio
from typing import cast

from pydantic import JsonValue

from harness.capabilities.rag import (
    AccessScope,
    DocumentCatalog,
    DocumentCatalogQuery,
)
from harness.logging import AgentLog
from harness.messages import ToolResult, ToolUse

log = AgentLog(__name__)


class DocumentCatalogTool:
    """查询已授权文档目录，不依赖语义 Top-K 结果进行计数。"""

    name = "document_catalog"
    description = (
        "List and count authorized knowledge-base documents using exact metadata filters. "
        "Use this for questions about how many documents, all matching files, or enumerating "
        "sources; use document_search for document content."
    )
    input_schema = DocumentCatalogQuery
    concurrency_group = None

    def __init__(self, catalog: DocumentCatalog, access_scope: AccessScope) -> None:
        self.catalog = catalog
        self.access_scope = access_scope

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        query = DocumentCatalogQuery.model_validate(tool_use.input)
        with log.operation(
            "rag.document_catalog",
            tool_name=self.name,
            tool_use_id=tool_use.id,
            has_source_filter=query.source_contains is not None,
            has_title_filter=query.title_contains is not None,
            filter_count=sum(
                value is not None
                for value in (query.source_contains, query.title_contains, query.category)
            )
            + bool(query.tags),
            limit=query.limit,
            include_public=self.access_scope.include_public,
            has_user_scope=self.access_scope.user_id is not None,
        ) as outcome:
            entries = await asyncio.to_thread(
                self.catalog.list_documents,
                query,
                self.access_scope,
            )
            documents = [
                {
                    "document_id": entry.document_id,
                    "source": entry.source,
                    "title": entry.title,
                    "scope": entry.scope,
                    "user_id": entry.user_id,
                    "chunk_count": entry.chunk_count,
                }
                for entry in entries
            ]
            outcome["document_count"] = len(documents)
            return ToolResult(
                tool_use_id=tool_use.id,
                content=cast(
                    JsonValue,
                    {
                        "total": len(documents),
                        "documents": documents,
                        "message": f"Found {len(documents)} authorized document(s).",
                    },
                ),
            )


__all__ = ["DocumentCatalogTool"]
