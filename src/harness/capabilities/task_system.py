"""任务创建、依赖、认领和完成能力。

Task creation, dependency, claim, and completion capability.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from harness.messages import ToolResult, ToolUse
from harness.permissions import PermissionDecision, PermissionResult
from harness.state import AgentState
from harness.tool_use import Tool, ToolExecutionContext, ToolInput

TASK_ID_PATTERN = r"^task_[0-9a-f]{32}$"
TASK_TOOL_NAMES = frozenset(
    {"create_task", "get_task", "list_tasks", "claim_task", "complete_task", "fail_task"}
)


class TaskStatus(StrEnum):
    """Task 允许的生命周期状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    """与具体 Store 无关的不可变 Task 记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=TASK_ID_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: tuple[str, ...] = ()
    owner: str | None = Field(default=None, min_length=1, max_length=128)
    result_reference: str | None = Field(default=None, min_length=1, max_length=1_000)
    failure_reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    run_id: str | None = Field(default=None, min_length=1, max_length=128)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def state_fields_must_be_consistent(self) -> TaskRecord:
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("task dependencies must be unique")
        if any(not _is_task_id(value) for value in self.dependencies):
            raise ValueError("task dependency has an invalid ID")
        if self.status is TaskStatus.PENDING and self.owner is not None:
            raise ValueError("pending task must not have an owner")
        if self.status is not TaskStatus.PENDING and self.owner is None:
            raise ValueError("claimed or terminal task requires an owner")
        if self.status is TaskStatus.COMPLETED and self.result_reference is None:
            raise ValueError("completed task requires a result reference")
        if self.status is TaskStatus.FAILED and self.failure_reason is None:
            raise ValueError("failed task requires a failure reason")
        if self.status is not TaskStatus.COMPLETED and self.result_reference is not None:
            raise ValueError("only completed task may have a result reference")
        if self.status is not TaskStatus.FAILED and self.failure_reason is not None:
            raise ValueError("only failed task may have a failure reason")
        return self


class TaskSystemError(RuntimeError):
    """Task System 领域错误基类。"""


class TaskNotFoundError(TaskSystemError):
    """Task ID 不存在。"""


class TaskDependencyError(TaskSystemError):
    """Task 依赖不存在或尚未完成。"""


class TaskTransitionError(TaskSystemError):
    """Task 状态转换或 Owner 校验失败。"""


@runtime_checkable
class TaskStore(Protocol):
    """持久化 Task Store 的最小同步契约。"""

    def create(
        self,
        title: str,
        description: str = "",
        dependencies: Sequence[str] = (),
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord: ...

    def list(self, status: TaskStatus | None = None) -> tuple[TaskRecord, ...]: ...

    def claim(self, task_id: str, owner: str) -> TaskRecord: ...

    def complete(self, task_id: str, owner: str, result_reference: str) -> TaskRecord: ...

    def fail(self, task_id: str, owner: str, failure_reason: str) -> TaskRecord: ...


class InMemoryTaskStore:
    """测试和未装配持久化服务时使用的线程安全 Task Store。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create(
        self,
        title: str,
        description: str = "",
        dependencies: Sequence[str] = (),
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> TaskRecord:
        with self._lock:
            normalized_dependencies = tuple(dependencies)
            _validate_dependencies_exist(normalized_dependencies, self._tasks)
            now = datetime.now(UTC)
            task = TaskRecord(
                task_id=_new_task_id(),
                title=title,
                description=description,
                dependencies=normalized_dependencies,
                conversation_id=conversation_id,
                run_id=run_id,
                created_at=now,
                updated_at=now,
            )
            self._tasks[task.task_id] = task
            return task

    def get(self, task_id: str) -> TaskRecord:
        with self._lock:
            return _require_task(task_id, self._tasks)

    def list(self, status: TaskStatus | None = None) -> tuple[TaskRecord, ...]:
        with self._lock:
            return tuple(
                task
                for task in self._tasks.values()
                if status is None or task.status is status
            )

    def claim(self, task_id: str, owner: str) -> TaskRecord:
        with self._lock:
            task = _require_task(task_id, self._tasks)
            _validate_claim(task, self._tasks)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.IN_PROGRESS,
                    "owner": owner,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._tasks[task_id] = updated
            return updated

    def complete(self, task_id: str, owner: str, result_reference: str) -> TaskRecord:
        return self._finish(task_id, owner, TaskStatus.COMPLETED, result_reference, None)

    def fail(self, task_id: str, owner: str, failure_reason: str) -> TaskRecord:
        return self._finish(task_id, owner, TaskStatus.FAILED, None, failure_reason)

    def _finish(
        self,
        task_id: str,
        owner: str,
        status: TaskStatus,
        result_reference: str | None,
        failure_reason: str | None,
    ) -> TaskRecord:
        with self._lock:
            task = _require_task(task_id, self._tasks)
            _validate_finish(task, owner)
            updated = task.model_copy(
                update={
                    "status": status,
                    "result_reference": result_reference,
                    "failure_reason": failure_reason,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._tasks[task_id] = updated
            return updated


class CreateTaskInput(ToolInput):
    """创建 Task 的输入。"""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    dependencies: tuple[str, ...] = ()

    @model_validator(mode="after")
    def dependencies_must_be_unique_and_valid(self) -> CreateTaskInput:
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("task dependencies must be unique")
        if any(not _is_task_id(value) for value in self.dependencies):
            raise ValueError("task dependency has an invalid ID")
        return self


class GetTaskInput(ToolInput):
    task_id: str = Field(pattern=TASK_ID_PATTERN)


class ListTasksInput(ToolInput):
    status: TaskStatus | None = None


class ClaimTaskInput(ToolInput):
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    owner: str = Field(min_length=1, max_length=128)


class CompleteTaskInput(ClaimTaskInput):
    result_reference: str = Field(min_length=1, max_length=1_000)


class FailTaskInput(ClaimTaskInput):
    failure_reason: str = Field(min_length=1, max_length=2_000)


class _TaskTool:
    """所有 Task Tool 共享的 Store 和串行组。"""

    concurrency_group = "task_store"

    def __init__(self, store: TaskStore) -> None:
        self.store = store

    @staticmethod
    def _result(tool_use: ToolUse, value: TaskRecord | Sequence[TaskRecord]) -> ToolResult:
        if isinstance(value, TaskRecord):
            content = value.model_dump(mode="json")
        else:
            content = [task.model_dump(mode="json") for task in value]
        return ToolResult(tool_use_id=tool_use.id, content=cast(JsonValue, content))


class CreateTaskTool(_TaskTool):
    name = "create_task"
    description = "Create a persistent task, optionally depending on existing tasks."
    input_schema = CreateTaskInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        return await self._create(tool_use)

    async def ainvoke_with_context(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """使用可信 Graph 上下文关联 Conversation 和 Run。"""

        run_id = context.metadata.get("run_id")
        return await self._create(
            tool_use,
            conversation_id=context.thread_id,
            run_id=run_id if isinstance(run_id, str) else None,
        )

    async def _create(
        self,
        tool_use: ToolUse,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> ToolResult:
        value = CreateTaskInput.model_validate(tool_use.input)
        task = self.store.create(
            value.title,
            value.description,
            value.dependencies,
            conversation_id,
            run_id,
        )
        return self._result(tool_use, task)


class GetTaskTool(_TaskTool):
    name = "get_task"
    description = "Read one persistent task by its exact task ID."
    input_schema = GetTaskInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = GetTaskInput.model_validate(tool_use.input)
        return self._result(tool_use, self.store.get(value.task_id))


class ListTasksTool(_TaskTool):
    name = "list_tasks"
    description = "List persistent tasks, optionally filtered by status."
    input_schema = ListTasksInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = ListTasksInput.model_validate(tool_use.input)
        return self._result(tool_use, self.store.list(value.status))


class ClaimTaskTool(_TaskTool):
    name = "claim_task"
    description = "Claim one pending task after all dependencies are completed."
    input_schema = ClaimTaskInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = ClaimTaskInput.model_validate(tool_use.input)
        return self._result(tool_use, self.store.claim(value.task_id, value.owner))


class CompleteTaskTool(_TaskTool):
    name = "complete_task"
    description = "Complete an owned in-progress task and save its result reference."
    input_schema = CompleteTaskInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = CompleteTaskInput.model_validate(tool_use.input)
        task = self.store.complete(value.task_id, value.owner, value.result_reference)
        return self._result(tool_use, task)


class FailTaskTool(_TaskTool):
    name = "fail_task"
    description = "Fail an owned in-progress task and save the failure reason."
    input_schema = FailTaskInput

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = FailTaskInput.model_validate(tool_use.input)
        task = self.store.fail(value.task_id, value.owner, value.failure_reason)
        return self._result(tool_use, task)


class TaskSystemPermissionRule:
    """允许修改当前用户隔离的内部 Task 元数据。"""

    name = "allow_task_system"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name not in TASK_TOOL_NAMES:
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="task tools only update the current user's internal task store",
        )


def create_task_tools(store: TaskStore) -> tuple[Tool, ...]:
    """创建使用同一个用户级 Store 的完整 Task Tool 集合。"""

    return (
        CreateTaskTool(store),
        GetTaskTool(store),
        ListTasksTool(store),
        ClaimTaskTool(store),
        CompleteTaskTool(store),
        FailTaskTool(store),
    )


def _new_task_id() -> str:
    return f"task_{uuid4().hex}"


def _is_task_id(value: str) -> bool:
    return value.startswith("task_") and len(value) == 37 and all(
        character in "0123456789abcdef" for character in value[5:]
    )


def _require_task(task_id: str, tasks: dict[str, TaskRecord]) -> TaskRecord:
    task = tasks.get(task_id)
    if task is None:
        raise TaskNotFoundError(f"unknown task: {task_id}")
    return task


def _validate_dependencies_exist(
    dependencies: Sequence[str],
    tasks: dict[str, TaskRecord],
) -> None:
    missing = [task_id for task_id in dependencies if task_id not in tasks]
    if missing:
        raise TaskDependencyError(f"unknown task dependencies: {', '.join(missing)}")


def _validate_claim(task: TaskRecord, tasks: dict[str, TaskRecord]) -> None:
    if task.status is not TaskStatus.PENDING:
        raise TaskTransitionError(f"task is not pending: {task.task_id}")
    blocked = [
        dependency
        for dependency in task.dependencies
        if _require_task(dependency, tasks).status is not TaskStatus.COMPLETED
    ]
    if blocked:
        raise TaskDependencyError(
            f"task dependencies are not completed: {', '.join(blocked)}"
        )


def _validate_finish(task: TaskRecord, owner: str) -> None:
    if task.status is not TaskStatus.IN_PROGRESS:
        raise TaskTransitionError(f"task is not in_progress: {task.task_id}")
    if task.owner != owner:
        raise TaskTransitionError(
            f"task owner mismatch: expected {task.owner}, got {owner}"
        )


__all__ = [
    "ClaimTaskInput",
    "ClaimTaskTool",
    "CompleteTaskInput",
    "CompleteTaskTool",
    "CreateTaskInput",
    "CreateTaskTool",
    "FailTaskInput",
    "FailTaskTool",
    "GetTaskInput",
    "GetTaskTool",
    "InMemoryTaskStore",
    "ListTasksInput",
    "ListTasksTool",
    "TASK_ID_PATTERN",
    "TASK_TOOL_NAMES",
    "TaskDependencyError",
    "TaskNotFoundError",
    "TaskRecord",
    "TaskStatus",
    "TaskStore",
    "TaskSystemError",
    "TaskSystemPermissionRule",
    "TaskTransitionError",
    "create_task_tools",
]
