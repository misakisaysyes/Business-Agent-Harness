"""Agent Bootstrap 装配测试。

Tests for agent bootstrap composition.
"""

from pathlib import Path
from typing import cast

import pytest
from tests.fakes import FakeModel, FakeSequenceModel

from entrypoints.bootstrap import UnknownModelConfigError, bootstrap_agent, resolve_model_settings
from harness.agent_loop import get_permission_request
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.profile import ModelConfigRef
from harness.state import AgentState
from services.config import AgentLoopSettings, AppSettings, RuntimePathSettings


def test_bootstrap_uses_injected_fake_model_without_external_sdk(tmp_path: Path) -> None:
    """注入 FakeModel 时 Bootstrap 不需要真实模型配置或网络。

    Bootstrap should need no real model configuration or network when FakeModel is injected.
    """

    response = Message(role=MessageRole.ASSISTANT, content="固定能力说明")
    fake_model = FakeModel(response)
    settings = AppSettings(
        paths=RuntimePathSettings(workspace_root=tmp_path),
        _env_file=None,
    )
    loop = bootstrap_agent(model=cast(ModelProvider, fake_model), settings=settings)
    state: AgentState = {
        "thread_id": "test-thread",
        "messages": [Message(role=MessageRole.USER, content="请介绍你能做什么")],
    }

    result = loop.invoke(state)

    assert result["messages"][-1] == response
    assert len(fake_model.sync_requests) == 1


def test_bootstrap_reads_from_configured_runtime_knowledge_root(tmp_path: Path) -> None:
    """Bootstrap 应从运行配置读取知识目录，不依赖模块安装路径。

    Bootstrap should use runtime paths instead of the module installation path.
    """

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "sample.txt").write_text("runtime sample", encoding="utf-8")
    settings = AppSettings(
        paths=RuntimePathSettings(
            workspace_root=workspace,
            knowledge_root=Path("knowledge"),
            artifact_root=Path("artifacts"),
        ),
        _env_file=None,
    )
    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(
                    ToolUse(
                        id="read-runtime-file",
                        name="file_reader",
                        input={"path": "sample.txt"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="读取完成"),
        ]
    )
    loop = bootstrap_agent(model=cast(ModelProvider, model), settings=settings)
    state: AgentState = {
        "thread_id": "runtime-path-thread",
        "messages": [Message(role=MessageRole.USER, content="读取 sample.txt")],
    }

    result = loop.invoke(state)

    tool_message = model.sync_requests[1].messages[-1]
    assert tool_message.role is MessageRole.TOOL
    assert tool_message.tool_results[0].content == "runtime sample"
    assert result["messages"][-1].content == "读取完成"


def test_bootstrap_asks_before_reading_other_workspace_file(tmp_path: Path) -> None:
    """知识目录外的工作区文件应在批准后才读取。

    A workspace file outside knowledge should be read only after approval.
    """

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    knowledge.mkdir(parents=True)
    (workspace / "private-note.txt").write_text("approved content", encoding="utf-8")
    settings = AppSettings(
        paths=RuntimePathSettings(
            workspace_root=workspace,
            knowledge_root=Path("knowledge"),
            artifact_root=Path("artifacts"),
        ),
        _env_file=None,
    )
    tool_use = ToolUse(
        id="read-workspace-file",
        name="file_reader",
        input={"path": "private-note.txt"},
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="读取完成"),
        ]
    )
    loop = bootstrap_agent(model=cast(ModelProvider, model), settings=settings)
    state: AgentState = {
        "thread_id": "workspace-approval-thread",
        "messages": [Message(role=MessageRole.USER, content="读取工作区文件")],
    }

    paused = loop.invoke(state)
    permission_request = get_permission_request(paused)

    assert permission_request is not None
    assert permission_request.requests[0].tool_use_id == tool_use.id
    assert len(model.sync_requests) == 1

    result = loop.resume("workspace-approval-thread", True)

    tool_message = model.sync_requests[1].messages[-1]
    assert tool_message.tool_results[0].content == "approved content"
    assert result["messages"][-1].content == "读取完成"


def test_bootstrap_uses_configured_agent_loop_iteration_limit(tmp_path: Path) -> None:
    """业务闭环应能把默认八轮以上的有界迭代配置传给 Agent Loop。"""

    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(
                    ToolUse(
                        id=f"calculate-{index}",
                        name="calculator",
                        input={"expression": f"{index} + 1"},
                    ),
                ),
            )
            for index in range(9)
        ]
        + [Message(role=MessageRole.ASSISTANT, content="长闭环完成")]
    )
    settings = AppSettings(
        paths=RuntimePathSettings(workspace_root=tmp_path),
        agent_loop=AgentLoopSettings(max_iterations=10),
        _env_file=None,
    )
    loop = bootstrap_agent(model=cast(ModelProvider, model), settings=settings)

    result = loop.invoke(
        {
            "thread_id": "configured-iterations",
            "messages": [Message(role=MessageRole.USER, content="执行长闭环")],
        }
    )

    assert result["messages"][-1].content == "长闭环完成"
    assert result["iteration_count"] == 10
    assert len(model.sync_requests) == 10


def test_unknown_model_configuration_fails_explicitly() -> None:
    """未知 ModelConfigRef 应产生明确错误。

    An unknown ModelConfigRef should fail explicitly.
    """

    with pytest.raises(UnknownModelConfigError, match="secondary"):
        resolve_model_settings(
            ModelConfigRef(name="secondary"),
            AppSettings(_env_file=None),
        )
