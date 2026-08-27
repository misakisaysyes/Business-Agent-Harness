"""通用 Agent 状态定义和 Reducer。

Shared agent state definitions and reducers.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, NotRequired, TypedDict

from pydantic import JsonValue

from harness.messages import Message, ToolResult, ToolUse


class AgentStopReason(StrEnum):
    """一次 Agent Turn 停止执行的标准原因。

    Standard reasons why an agent turn stopped executing.
    """

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    CANCELLED = "cancelled"
    HOOK_BLOCKED = "hook_blocked"


def append_messages(
    current: Sequence[Message] | None,
    updates: Sequence[Message] | None,
) -> list[Message]:
    """返回追加消息后的新列表，不修改任何输入列表。

    Return a new list with appended messages without mutating either input.
    """

    current_messages = list(current) if current else []
    new_messages = list(updates) if updates else []
    return current_messages + new_messages


class AgentState(TypedDict):
    """所有业务 Agent 共享的最小运行状态。

    Minimal runtime state shared by all business agents.
    """

    thread_id: str
    messages: Annotated[list[Message], append_messages]
    metadata: NotRequired[dict[str, JsonValue]]
    iteration_count: NotRequired[int]
    cancel_requested: NotRequired[bool]
    stop_reason: NotRequired[AgentStopReason | None]
    pending_tool_uses: NotRequired[list[ToolUse]]
    pending_tool_results: NotRequired[list[ToolResult]]
    capability_state: NotRequired[dict[str, JsonValue]]


class AgentStateUpdate(TypedDict, total=False):
    """LangGraph Node 可以返回的增量状态。

    Partial state update that a LangGraph node may return.
    """

    thread_id: str
    messages: list[Message]
    metadata: dict[str, JsonValue]
    iteration_count: int
    cancel_requested: bool
    stop_reason: AgentStopReason | None
    pending_tool_uses: list[ToolUse]
    pending_tool_results: list[ToolResult]
    capability_state: dict[str, JsonValue]


__all__ = ["AgentState", "AgentStateUpdate", "AgentStopReason", "append_messages"]
