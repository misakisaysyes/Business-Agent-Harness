"""Multi-Agent 子 Agent 的任务、上下文和结果契约。

Contracts for isolated Multi-Agent subagent tasks, contexts, and results.
"""

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class TaskStatus(StrEnum):
    """Team 任务生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# M7 兼容用别名；业务代码可逐步迁移到 TaskStatus。
SubagentStatus = TaskStatus


class AgentRole(BaseModel):
    """角色的业务无关描述和工具能力边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1)
    allowed_tool_names: tuple[str, ...] = ()


class DelegationBudget(BaseModel):
    """一次 Team Run 允许消耗的委派预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tasks: int = Field(default=8, ge=1)
    max_depth: int = Field(default=1, ge=0)
    max_iterations: int = Field(default=8, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_duration_seconds: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=0, ge=0)
    max_context_chars: int = Field(default=12_000, ge=1)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0.0)
    retry_max_delay_seconds: float = Field(default=4.0, ge=0.0)
    retry_jitter_seconds: float = Field(default=0.1, ge=0.0)

class SubagentContext(BaseModel):
    """传给子 Agent 的可信运行时上下文；主会话消息不会自动透传。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_thread_id: str = Field(min_length=1)
    parent_run_id: str | None = None
    team_run_id: str | None = None
    user_id: str | None = None
    search_mode: str = Field(default="auto", min_length=1)
    depth: int = Field(default=0, ge=0)
    allowed_tool_names: tuple[str, ...] = ()
    budget: DelegationBudget = Field(default_factory=DelegationBudget)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TeamTask(BaseModel):
    """一次可追踪、可独立失败的 Team 子任务。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex}", min_length=1)
    parent_task_id: str | None = None
    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    objective: str = Field(min_length=1, max_length=12_000)
    input: dict[str, JsonValue] = Field(default_factory=dict)
    context: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_tool_names: tuple[str, ...] = ()
    budget: DelegationBudget = Field(default_factory=DelegationBudget)


class TeamTaskResult(BaseModel):
    """子 Agent 返回给 Lead 的结构化任务结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    status: TaskStatus
    summary: str = ""
    evidence: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    child_thread_id: str | None = None
    error_reason: str | None = None
    tool_names: tuple[str, ...] = ()
    attempt: int = Field(default=1, ge=1)
    attempts: int = Field(default=1, ge=1)
    last_error: str | None = None

    @property
    def succeeded(self) -> bool:
        """返回任务是否成功完成。"""

        return self.status is TaskStatus.SUCCEEDED


class TeamRun(BaseModel):
    """一次 Team 执行的可审计元数据和预算快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team_run_id: str = Field(min_length=1)
    parent_run_id: str | None = None
    status: TaskStatus = TaskStatus.RUNNING
    budget: DelegationBudget = Field(default_factory=DelegationBudget)


class SubagentDefinition(BaseModel):
    """一个业务角色的最小运行定义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    system_prompt: str = Field(min_length=1)
    description: str = Field(default="", max_length=2_000)
    allowed_tool_names: tuple[str, ...] = ()
    max_iterations: int = Field(default=8, ge=1)

    @field_validator("allowed_tool_names")
    @classmethod
    def validate_tool_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("allowed tool names must be unique")
        if any(not name.strip() for name in value):
            raise ValueError("allowed tool names must not be blank")
        return value


# 旧内部命名保留，避免 M7 期间的调用方发生破坏性变更。
SubagentTask = TeamTask
SubagentResult = TeamTaskResult


__all__ = [
    "AgentRole",
    "DelegationBudget",
    "SubagentContext",
    "SubagentDefinition",
    "SubagentResult",
    "SubagentStatus",
    "SubagentTask",
    "TaskStatus",
    "TeamRun",
    "TeamTask",
    "TeamTaskResult",
]
