"""Knowledge Assistant 业务 Tool 测试。

Tests for Knowledge Assistant business tools.
"""

from pathlib import Path

import pytest

from business.knowledge_assistant.tools import CalculatorTool, FileReaderTool, ReportWriterTool
from harness.messages import ToolUse
from harness.tool_use import ToolRegistry
from services.artifacts import ArtifactStore


def test_file_tools_share_one_concurrency_group(tmp_path: Path) -> None:
    """文件读写 Tool 不依赖目录差异判断并发安全性。

    File tools must not rely on directory separation for concurrency safety.
    """

    reader = FileReaderTool((tmp_path,))
    writer = ReportWriterTool(ArtifactStore(tmp_path))

    assert reader.concurrency_group == "filesystem"
    assert writer.concurrency_group == reader.concurrency_group


@pytest.mark.asyncio
async def test_calculator_supports_arithmetic_and_rejects_code_execution() -> None:
    """Calculator 应正确计算算术并拒绝函数调用等代码。

    Calculator should evaluate arithmetic and reject function calls or other code.
    """

    registry = ToolRegistry((CalculatorTool(),))
    success = await registry.dispatch(
        ToolUse(id="calc-1", name="calculator", input={"expression": "2 + 3 * 4"})
    )
    blocked = await registry.dispatch(
        ToolUse(
            id="calc-2",
            name="calculator",
            input={"expression": "__import__('os').system('echo unsafe')"},
        )
    )

    assert success.content == 14
    assert not success.is_error
    assert blocked.is_error
    assert "unsupported expression" in str(blocked.content)


@pytest.mark.asyncio
async def test_file_reader_allows_authorized_file_and_blocks_traversal(tmp_path: Path) -> None:
    """File Reader 应读取授权文件并阻止路径穿越。

    File Reader should read authorized files and block path traversal.
    """

    allowed_root = tmp_path / "knowledge"
    allowed_root.mkdir()
    (allowed_root / "note.txt").write_text("authorized", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    registry = ToolRegistry((FileReaderTool((allowed_root,)),))

    success = await registry.dispatch(
        ToolUse(id="read-1", name="file_reader", input={"path": "note.txt"})
    )
    blocked = await registry.dispatch(
        ToolUse(id="read-2", name="file_reader", input={"path": "../secret.txt"})
    )

    assert success.content == "authorized"
    assert not success.is_error
    assert blocked.is_error
    assert "PermissionError" in str(blocked.content)


@pytest.mark.asyncio
async def test_file_reader_resolves_relative_paths_from_knowledge_then_workspace(
    tmp_path: Path,
) -> None:
    """相对路径应先查知识目录，再查工作区。

    Relative paths should search the knowledge directory before the workspace.
    """

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "sample.txt").write_text("knowledge sample", encoding="utf-8")
    (workspace / "workspace.txt").write_text("workspace sample", encoding="utf-8")
    registry = ToolRegistry((FileReaderTool((workspace,), default_root=knowledge),))

    knowledge_result = await registry.dispatch(
        ToolUse(id="read-knowledge", name="file_reader", input={"path": "sample.txt"})
    )
    workspace_result = await registry.dispatch(
        ToolUse(id="read-workspace", name="file_reader", input={"path": "workspace.txt"})
    )

    assert knowledge_result.content == "knowledge sample"
    assert workspace_result.content == "workspace sample"


@pytest.mark.asyncio
async def test_report_writer_prevents_default_overwrite_and_rejects_path_escape(
    tmp_path: Path,
) -> None:
    """Report Writer 默认不得覆盖，并且不得逃逸 Artifact 根目录。

    Report Writer should prevent default overwrites and reject artifact path escape.
    """

    artifact_root = tmp_path / "artifacts"
    registry = ToolRegistry((ReportWriterTool(ArtifactStore(artifact_root)),))

    created = await registry.dispatch(
        ToolUse(
            id="write-1",
            name="report_writer",
            input={"path": "reports/result.md", "content": "first"},
        )
    )
    duplicate = await registry.dispatch(
        ToolUse(
            id="write-2",
            name="report_writer",
            input={"path": "reports/result.md", "content": "second"},
        )
    )
    escaped = await registry.dispatch(
        ToolUse(
            id="write-3",
            name="report_writer",
            input={"path": "../outside.md", "content": "unsafe"},
        )
    )

    assert not created.is_error
    assert (artifact_root / "reports/result.md").read_text(encoding="utf-8") == "first"
    assert duplicate.is_error
    assert "FileExistsError" in str(duplicate.content)
    assert escaped.is_error
    assert "InvalidArtifactPathError" in str(escaped.content)
