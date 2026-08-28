"""调用共享 RAG Pipeline 的业务 Tool。

Business-facing tool for invoking the shared RAG pipeline.
"""

import asyncio
from typing import cast

from pydantic import Field, JsonValue

from harness.capabilities.rag import AccessScope, RAGPipeline, RetrievalQuery
from harness.logging import AgentLog
from harness.messages import ToolResult, ToolUse
from harness.tool_use import ToolInput

log = AgentLog(__name__)


class DocumentSearchInput(ToolInput):
    """模型可填写的检索参数；不包含任何身份或数据库字段。"""

    query: str = Field(min_length=1, max_length=8_000)
    category: str | None = Field(default=None, min_length=1, max_length=128)
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=20)


class DocumentSearchTool:
    """绑定可信 AccessScope 的只读知识库检索 Tool。"""

    name = "document_search"
    description = (
        "Search authorized public and private knowledge-base documents. "
        "Use returned snippets and citation IDs as evidence."
    )
    input_schema = DocumentSearchInput
    concurrency_group = None

    def __init__(
        self,
        pipeline: RAGPipeline,
        access_scope: AccessScope,
        *,
        default_top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> None:
        self.pipeline = pipeline
        self.access_scope = access_scope
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        tool_input = DocumentSearchInput.model_validate(tool_use.input)
        filters: dict[str, JsonValue] = {}
        if tool_input.category is not None:
            filters["category"] = tool_input.category
        if tool_input.tags:
            filters["tags"] = list(tool_input.tags)
        top_k = tool_input.top_k or self.default_top_k
        with log.operation(
            "rag.document_search",
            tool_name=self.name,
            tool_use_id=tool_use.id,
            query_characters=len(tool_input.query),
            top_k=top_k,
            score_threshold=self.score_threshold,
            filter_count=len(filters),
            include_public=self.access_scope.include_public,
            has_user_scope=self.access_scope.user_id is not None,
        ) as outcome:
            result = await asyncio.to_thread(
                self.pipeline.search,
                RetrievalQuery(
                    text=tool_input.query,
                    access_scope=self.access_scope,
                    top_k=top_k,
                    score_threshold=self.score_threshold,
                    filters=filters,
                ),
            )
            matches: list[dict[str, JsonValue]] = []
            for hit, citation in zip(result.matches, result.citations, strict=True):
                matches.append(
                    {
                        "citation": f"[{citation.id}]",
                        "source": citation.source,
                        "section": citation.section,
                        "chunk_id": citation.chunk_id,
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                )
            outcome["selected_count"] = len(matches)
            outcome["context_characters"] = len(result.context)
            if not matches:
                outcome["status"] = "no_matches"
            return ToolResult(
                tool_use_id=tool_use.id,
                content=cast(
                    JsonValue,
                    {
                        "message": result.message,
                        "matches": matches,
                        "context": result.context,
                    },
                ),
            )


__all__ = ["DocumentSearchInput", "DocumentSearchTool"]
