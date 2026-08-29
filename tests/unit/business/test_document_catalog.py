"""Knowledge Assistant document_catalog Tool 测试。"""

import pytest

from business.knowledge_assistant.tools import DocumentCatalogTool
from harness.capabilities.rag import (
    AccessScope,
    DocumentCatalogEntry,
    DocumentCatalogQuery,
)
from harness.messages import ToolUse


class FakeCatalog:
    def __init__(self) -> None:
        self.queries: list[tuple[DocumentCatalogQuery, AccessScope]] = []

    def list_documents(
        self,
        query: DocumentCatalogQuery,
        access_scope: AccessScope,
    ) -> tuple[DocumentCatalogEntry, ...]:
        self.queries.append((query, access_scope))
        return (
            DocumentCatalogEntry(
                document_id="doc-1",
                source="字节广告一面_原文.docx",
                title="字节广告一面",
                scope="public",
                chunk_count=3,
            ),
            DocumentCatalogEntry(
                document_id="doc-2",
                source="字节广告二面_原文.docx",
                title="字节广告二面",
                scope="public",
                chunk_count=2,
            ),
        )


@pytest.mark.asyncio
async def test_document_catalog_returns_unique_authorized_documents() -> None:
    catalog = FakeCatalog()
    tool = DocumentCatalogTool(catalog, AccessScope(user_id="alice"))

    result = await tool.ainvoke(
        ToolUse(
            id="catalog-1",
            name="document_catalog",
            input={"source_contains": "字节", "limit": 20},
        )
    )

    assert result.tool_use_id == "catalog-1"
    assert result.content["total"] == 2  # type: ignore[index]
    assert [item["source"] for item in result.content["documents"]] == [  # type: ignore[index]
        "字节广告一面_原文.docx",
        "字节广告二面_原文.docx",
    ]
    assert catalog.queries[0][0].source_contains == "字节"
    assert catalog.queries[0][1].user_id == "alice"


@pytest.mark.asyncio
async def test_document_catalog_rejects_model_supplied_identity() -> None:
    tool = DocumentCatalogTool(FakeCatalog(), AccessScope(user_id="alice"))

    with pytest.raises(ValueError):
        await tool.ainvoke(
            ToolUse(
                id="catalog-2",
                name="document_catalog",
                input={"user_id": "bob"},
            )
        )
