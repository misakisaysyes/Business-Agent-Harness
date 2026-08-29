"""Knowledge Assistant Permission Rule 测试。

Tests for Knowledge Assistant permission rules.
"""

from pathlib import Path

from business.knowledge_assistant.permission_rules import (
    CalculatorPermissionRule,
    ExternalPublishPermissionRule,
    FileReadPermissionRule,
    ReportWritePermissionRule,
    SearchModePermissionRule,
)
from business.knowledge_assistant.tools import FileReaderTool
from harness.messages import Message, MessageRole, ToolUse
from harness.permissions import PermissionDecision, PermissionPipeline
from harness.state import AgentState
from services.artifacts import ArtifactStore


def state() -> AgentState:
    """创建业务权限测试状态。

    Create state used by business permission tests.
    """

    return {
        "thread_id": "business-permission-thread",
        "messages": [Message(role=MessageRole.USER, content="test")],
    }


async def test_file_read_allows_knowledge_asks_workspace_and_denies_escape(
    tmp_path: Path,
) -> None:
    """文件读取规则应允许知识目录、询问工作区并拒绝路径逃逸。

    The rule should allow knowledge, ask for workspace, and deny path escape.
    """

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "note.txt").write_text("knowledge", encoding="utf-8")
    (workspace / "workspace-note.txt").write_text("workspace", encoding="utf-8")

    reader = FileReaderTool((workspace,), default_root=knowledge)
    pipeline = PermissionPipeline(
        (FileReadPermissionRule(reader, auto_allowed_roots=(knowledge,)),),
        known_tool_names=(reader.name,),
    )

    allowed = await pipeline.evaluate(
        ToolUse(id="read-1", name="file_reader", input={"path": "note.txt"}),
        state(),
    )
    requires_approval = await pipeline.evaluate(
        ToolUse(
            id="read-2",
            name="file_reader",
            input={"path": "workspace-note.txt"},
        ),
        state(),
    )
    denied = await pipeline.evaluate(
        ToolUse(
            id="read-3",
            name="file_reader",
            input={"path": str(tmp_path / "outside.txt")},
        ),
        state(),
    )

    assert allowed.decision is PermissionDecision.ALLOW
    assert requires_approval.decision is PermissionDecision.ASK
    assert denied.decision is PermissionDecision.DENY


async def test_report_write_allows_create_asks_overwrite_and_denies_escape(
    tmp_path: Path,
) -> None:
    """报告规则应允许新建、询问覆盖并拒绝路径逃逸。

    The report rule should allow creation, ask on overwrite, and deny path escape.
    """

    store = ArtifactStore(tmp_path / "artifacts")
    pipeline = PermissionPipeline(
        (ReportWritePermissionRule(store),),
        known_tool_names=("report_writer",),
    )

    created = await pipeline.evaluate(
        ToolUse(
            id="write-1",
            name="report_writer",
            input={"path": "report.md", "content": "draft"},
        ),
        state(),
    )
    overwrite = await pipeline.evaluate(
        ToolUse(
            id="write-2",
            name="report_writer",
            input={"path": "report.md", "content": "final", "overwrite": True},
        ),
        state(),
    )
    escaped = await pipeline.evaluate(
        ToolUse(
            id="write-3",
            name="report_writer",
            input={"path": "../outside.md", "content": "unsafe"},
        ),
        state(),
    )

    assert created.decision is PermissionDecision.ALLOW
    assert overwrite.decision is PermissionDecision.ASK
    assert escaped.decision is PermissionDecision.DENY


async def test_calculator_is_allowed_and_external_publish_is_denied() -> None:
    """Calculator 应允许，外部发布应默认拒绝。

    Calculator should be allowed and external publishing denied by default.
    """

    pipeline = PermissionPipeline(
        (CalculatorPermissionRule(), ExternalPublishPermissionRule()),
        known_tool_names=("calculator",),
    )

    calculator = await pipeline.evaluate(ToolUse(id="calc", name="calculator"), state())
    publish = await pipeline.evaluate(ToolUse(id="publish", name="external_publish"), state())

    assert calculator.decision is PermissionDecision.ALLOW
    assert publish.decision is PermissionDecision.DENY


async def test_search_mode_denies_the_opposite_search_source() -> None:
    pipeline = PermissionPipeline(
        (SearchModePermissionRule(),),
        known_tool_names=("document_search", "web_search"),
    )
    rag_state = {**state(), "metadata": {"search_mode": "rag"}}
    web_state = {**state(), "metadata": {"search_mode": "web"}}

    rag_denied = await pipeline.evaluate(
        ToolUse(id="web-1", name="web_search"),
        rag_state,
    )
    web_denied = await pipeline.evaluate(
        ToolUse(id="rag-1", name="document_search"),
        web_state,
    )

    assert rag_denied.decision is PermissionDecision.DENY
    assert web_denied.decision is PermissionDecision.DENY
