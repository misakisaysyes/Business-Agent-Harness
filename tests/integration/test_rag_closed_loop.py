"""FakeModel 驱动的 document_search 单 Agent 闭环。"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tests.fakes import FakeSequenceModel

from entrypoints.bootstrap import bootstrap_agent
from harness.capabilities.rag import (
    DocumentChunk,
    RAGPipeline,
    RetrievalHit,
    RetrievalQuery,
)
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.state import AgentState
from services.config import AppSettings, RuntimePathSettings


class ScopedRetriever:
    def __init__(self) -> None:
        self.queries: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        self.queries.append(query)
        return (
            RetrievalHit(
                chunk=DocumentChunk(
                    document_id="refund-guide",
                    chunk_id="refund-seven-days",
                    text="退款申请必须在订单完成后七天内提交。",
                    metadata={
                        "source": "product-guide.md",
                        "section": "退款规则",
                        "scope": "public",
                    },
                ),
                score=0.95,
                rank=1,
            ),
        )


def test_fake_model_calls_document_search_and_answers_with_real_citation(
    tmp_path: Path,
) -> None:
    retriever = ScopedRetriever()
    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(
                    ToolUse(
                        id="search-refund",
                        name="document_search",
                        input={"query": "退款期限"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="退款需在七天内申请。[S1]"),
        ]
    )
    settings = AppSettings(
        paths=RuntimePathSettings(workspace_root=tmp_path),
        _env_file=None,
    )
    loop = bootstrap_agent(
        model=cast(ModelProvider, model),
        settings=settings,
        rag_pipeline=RAGPipeline(retriever),
        trusted_user_id="alice",
    )
    state: AgentState = {
        "thread_id": "rag-closed-loop",
        "messages": [Message(role=MessageRole.USER, content="退款期限是多少？")],
    }

    result = loop.invoke(state)

    assert result["messages"][-1].content == "退款需在七天内申请。[S1]"
    tool_result = model.sync_requests[1].messages[-1].tool_results[0]
    assert "product-guide.md" in str(tool_result.content)
    assert retriever.queries[0].access_scope.user_id == "alice"
