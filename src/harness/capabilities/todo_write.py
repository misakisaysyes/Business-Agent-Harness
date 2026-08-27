"""TodoWrite 计划管理能力。

TodoWrite planning capability.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.messages import ToolResult, ToolUse
from harness.permissions import PermissionDecision, PermissionResult
from harness.state import AgentState
from harness.tool_use import ToolInput, ToolStatePatch

TODO_STATE_NAMESPACE = "todo_write"
MAX_TODO_ITEMS = 20


class TodoStatus(StrEnum):
    """Todo 项目的执行状态。

    Execution status of a todo item.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoItem(BaseModel):
    """一条不可变 Todo 项目。

    One immutable todo item.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1, max_length=500)
    status: TodoStatus = TodoStatus.PENDING


class TodoWriteInput(ToolInput):
    """TodoWrite 每次提交的完整清单。

    Complete todo list submitted on each TodoWrite call.
    """

    todos: tuple[TodoItem, ...] = Field(max_length=MAX_TODO_ITEMS)

    @model_validator(mode="after")
    def allow_one_current_step(self) -> "TodoWriteInput":
        """同一时间最多允许一个 in_progress Todo。

        Allow at most one in-progress todo at a time.
        """

        active_count = sum(item.status is TodoStatus.IN_PROGRESS for item in self.todos)
        if active_count > 1:
            raise ValueError("only one todo can be in_progress at a time")
        return self


class TodoSnapshot(BaseModel):
    """保存到当前 Thread Capability State 的 Todo 快照。

    Todo snapshot stored in the current thread's capability state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[TodoItem, ...] = ()
    current_step: str | None = None

    @classmethod
    def from_items(cls, items: tuple[TodoItem, ...]) -> "TodoSnapshot":
        current_step = next(
            (item.content for item in items if item.status is TodoStatus.IN_PROGRESS),
            None,
        )
        return cls(items=items, current_step=current_step)


class TodoWriteTool:
    """整表替换当前 Conversation Todo 快照的状态型 Tool。

    Stateful tool that replaces the current conversation's todo snapshot.
    """

    name = "todo_write"
    description = (
        "Create or replace the current multi-step plan. Submit the complete list on every "
        "call and keep at most one item in_progress."
    )
    input_schema = TodoWriteInput
    concurrency_group = "agent_state"

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        validated = TodoWriteInput.model_validate(tool_use.input)
        snapshot = TodoSnapshot.from_items(validated.todos)
        return ToolResult(
            tool_use_id=tool_use.id,
            content=snapshot.model_dump(mode="json"),
        )

    def state_update(
        self,
        tool_use: ToolUse,
        tool_result: ToolResult,
    ) -> ToolStatePatch:
        """把成功执行的清单写入 TodoWrite 命名空间。

        Write the successful list into the TodoWrite state namespace.
        """

        validated = TodoWriteInput.model_validate(tool_use.input)
        snapshot = TodoSnapshot.from_items(validated.todos)
        return ToolStatePatch(
            namespace=TODO_STATE_NAMESPACE,
            value=snapshot.model_dump(mode="json"),
        )


class TodoWritePermissionRule:
    """允许只修改当前 Agent State 的 TodoWrite Tool。

    Allow TodoWrite because it only updates the current agent state.
    """

    name = "allow_todo_write"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != TodoWriteTool.name:
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="todo_write only updates the current conversation state",
        )


__all__ = [
    "MAX_TODO_ITEMS",
    "TODO_STATE_NAMESPACE",
    "TodoItem",
    "TodoSnapshot",
    "TodoStatus",
    "TodoWriteInput",
    "TodoWritePermissionRule",
    "TodoWriteTool",
]
