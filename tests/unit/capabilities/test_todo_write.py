"""TodoWrite 状态和 Agent Loop 集成测试。

TodoWrite state and agent-loop integration tests.
"""

from typing import cast

import pytest
from pydantic import ValidationError
from tests.fakes import FakeSequenceModel

from harness.agent_loop import create_agent_loop
from harness.capabilities.todo_write import (
    TodoWriteInput,
    TodoWritePermissionRule,
    TodoWriteTool,
)
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.state import AgentState
from services.checkpoint import create_in_memory_checkpointer


def system_prompt() -> str:
    """返回 TodoWrite 测试 Prompt。"""

    return "Use TodoWrite for multi-step tasks."


def test_todo_write_rejects_multiple_current_steps() -> None:
    """同一份计划不能包含多个进行中步骤。"""

    with pytest.raises(ValidationError, match="only one todo"):
        TodoWriteInput.model_validate(
            {
                "todos": [
                    {"content": "读取资料", "status": "in_progress"},
                    {"content": "生成摘要", "status": "in_progress"},
                ]
            }
        )


def test_todo_write_persists_snapshot_in_thread_state() -> None:
    """成功的 TodoWrite 应把完整快照保存到当前 Thread State。"""

    tool_use = ToolUse(
        id="todo-001",
        name="todo_write",
        input={
            "todos": [
                {"content": "读取资料", "status": "completed"},
                {"content": "生成摘要", "status": "in_progress"},
            ]
        },
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="计划已更新"),
            Message(role=MessageRole.ASSISTANT, content="继续执行"),
            Message(role=MessageRole.ASSISTANT, content="新的会话"),
        ]
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(TodoWriteTool(),),
        permission_rules=(TodoWritePermissionRule(),),
    )
    state: AgentState = {
        "thread_id": "todo-thread",
        "messages": [Message(role=MessageRole.USER, content="整理资料")],
    }

    result = loop.invoke(state)

    todo_state = cast(dict[str, object], result["capability_state"]["todo_write"])
    assert todo_state["current_step"] == "生成摘要"
    assert todo_state["items"] == [
        {"content": "读取资料", "status": "completed"},
        {"content": "生成摘要", "status": "in_progress"},
    ]
    assert "capability_state" not in state

    continued = loop.invoke(
        {
            "thread_id": "todo-thread",
            "messages": [Message(role=MessageRole.USER, content="继续")],
        }
    )
    isolated = loop.invoke(
        {
            "thread_id": "another-thread",
            "messages": [Message(role=MessageRole.USER, content="新任务")],
        }
    )

    assert continued["capability_state"]["todo_write"] == todo_state
    assert "capability_state" not in isolated
