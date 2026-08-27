"""Knowledge Assistant document_search Tool 测试。"""

from collections.abc import Sequence

import pytest

from business.knowledge_assistant.tools import DocumentSearchTool
from harness.capabilities.rag import (
    AccessScope,
    DocumentChunk,
    RAGPipeline,
    RetrievalHit,
    RetrievalQuery,
)
from harness.messages import ToolUse
from harness.tool_use import ToolRegistry


class FixedRetriever:
    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        return (
            RetrievalHit(
                chunk=DocumentChunk(
                    document_id="doc",
                    chunk_id="chunk",
                    text="Refunds are accepted within seven days.",
                    metadata={
                        "source": "guide.md",
                        "section": "Refunds",
                        "scope": "public",
                    },
                ),
                score=0.9,
                rank=1,
            ),
        )


@pytest.mark.asyncio
async def test_document_search_returns_sanitized_snippet_and_real_citation() -> None:
    tool = DocumentSearchTool(RAGPipeline(FixedRetriever()), AccessScope(user_id="alice"))
    result = await ToolRegistry((tool,)).dispatch(
        ToolUse(id="search-1", name="document_search", input={"query": "refund"})
    )

    assert not result.is_error
    assert isinstance(result.content, dict)
    matches = result.content["matches"]
    assert isinstance(matches, list)
    assert matches[0]["citation"] == "[S1]"
    assert matches[0]["source"] == "guide.md"
    assert "database" not in str(result.content).casefold()


@pytest.mark.asyncio
async def test_document_search_rejects_model_supplied_identity() -> None:
    tool = DocumentSearchTool(RAGPipeline(FixedRetriever()), AccessScope(user_id="alice"))
    result = await ToolRegistry((tool,)).dispatch(
        ToolUse(
            id="search-identity",
            name="document_search",
            input={"query": "secret", "user_id": "bob"},
        )
    )

    assert result.is_error
    assert "invalid_tool_input" in str(result.content)
