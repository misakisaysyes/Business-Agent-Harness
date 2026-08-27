"""Hook 事件、注册和分发。

Hook events, registration, and dispatch.
"""

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.messages import Message, ToolResult, ToolUse
from harness.permissions import PermissionResult
from harness.state import AgentState, AgentStopReason

DEFAULT_LARGE_OUTPUT_CHARS = 8_000
logger = logging.getLogger(__name__)


class HookEventType(StrEnum):
    """与 learn-claude-code 对齐的 Hook 事件名称。

    Hook event names aligned with learn-claude-code.
    """

    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"


class HookDecision(StrEnum):
    """Hook 对当前操作的处理结果。

    Hook decision for the current operation.
    """

    CONTINUE = "continue"
    BLOCK = "block"


class HookFailureMode(StrEnum):
    """Hook 抛出异常时的处理方式。

    Behavior when a hook raises an exception.
    """

    CONTINUE = "continue"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class UserPromptSubmit:
    """提交用户消息后、调用模型前触发的事件。

    Event emitted after user submission and before model invocation.
    """

    event_type: ClassVar[HookEventType] = HookEventType.USER_PROMPT_SUBMIT

    state: AgentState
    message: Message


@dataclass(frozen=True, slots=True)
class PreToolUse:
    """Permission 评估后、Tool 执行前触发的事件。

    Event emitted after permission evaluation and before tool execution.
    """

    event_type: ClassVar[HookEventType] = HookEventType.PRE_TOOL_USE

    state: AgentState
    tool_use: ToolUse
    permission_result: PermissionResult
    permission_granted: bool


@dataclass(frozen=True, slots=True)
class PostToolUse:
    """Tool 返回结果后触发的只读观察事件。

    Read-only observation event emitted after a tool returns its result.
    """

    event_type: ClassVar[HookEventType] = HookEventType.POST_TOOL_USE

    state: AgentState
    tool_use: ToolUse
    tool_result: ToolResult


@dataclass(frozen=True, slots=True)
class Stop:
    """Agent Turn 进入 END 前触发的事件。

    Event emitted before an agent turn reaches END.
    """

    event_type: ClassVar[HookEventType] = HookEventType.STOP

    state: AgentState
    reason: AgentStopReason


HookEvent = UserPromptSubmit | PreToolUse | PostToolUse | Stop


class HookResult(BaseModel):
    """一个 Hook 的标准返回结果。

    Standard result returned by one hook.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: HookDecision = HookDecision.CONTINUE
    reason: str | None = None

    @model_validator(mode="after")
    def require_block_reason(self) -> "HookResult":
        """阻断操作时必须提供可返回给模型或用户的原因。

        Require a reason when blocking an operation.
        """

        if self.decision is HookDecision.BLOCK and not self.reason:
            raise ValueError("a blocking hook result requires a reason")
        return self


class HookError(BaseModel):
    """一次被 Hook Registry 捕获的异常记录。

    Error record captured by the hook registry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hook_name: str = Field(min_length=1)
    event_type: HookEventType
    error_type: str = Field(min_length=1)


class HookDispatchResult(BaseModel):
    """一轮 Hook 分发的聚合结果。

    Aggregate result of one hook dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: HookDecision = HookDecision.CONTINUE
    reason: str | None = None
    hook_name: str | None = None
    errors: tuple[HookError, ...] = ()

    @property
    def blocked(self) -> bool:
        """返回是否有 Hook 阻断了当前操作。

        Return whether a hook blocked the current operation.
        """

        return self.decision is HookDecision.BLOCK


@runtime_checkable
class Hook(Protocol):
    """所有 Hook 实现遵循的异步协议。

    Asynchronous protocol implemented by all hooks.
    """

    @property
    def name(self) -> str:
        """返回 Hook 的唯一名称。

        Return the unique hook name.
        """
        ...

    @property
    def events(self) -> frozenset[HookEventType]:
        """返回该 Hook 订阅的事件集合。

        Return the event types subscribed to by this hook.
        """
        ...

    async def handle(self, event: HookEvent) -> HookResult | None:
        """处理一个 Hook 事件。

        Handle one hook event.
        """
        ...


class DuplicateHookError(ValueError):
    """Hook Registry 中出现重复名称。

    Raised when duplicate names are registered in a hook registry.
    """


class HookExecutionError(RuntimeError):
    """配置为 RAISE 的 Hook 执行失败。

    Raised when a hook fails under RAISE failure mode.
    """


class HookRegistry:
    """按注册顺序分发 Hook，并统一处理阻断和异常。

    Dispatch hooks in registration order and centralize blocking and error handling.
    """

    def __init__(
        self,
        hooks: Sequence[Hook] = (),
        failure_mode: HookFailureMode = HookFailureMode.CONTINUE,
    ) -> None:
        self.failure_mode = failure_mode
        self._hooks: list[Hook] = []
        self._names: set[str] = set()
        for hook in hooks:
            self.register(hook)

    def register(self, hook: Hook) -> None:
        """按顺序注册 Hook，并拒绝重复名称。

        Register a hook in order and reject duplicate names.
        """

        if hook.name in self._names:
            raise DuplicateHookError(f"hook is already registered: {hook.name}")
        self._hooks.append(hook)
        self._names.add(hook.name)

    def names(self) -> tuple[str, ...]:
        """返回 Hook 注册顺序。

        Return hook names in registration order.
        """

        return tuple(hook.name for hook in self._hooks)

    async def dispatch(self, event: HookEvent) -> HookDispatchResult:
        """依次执行订阅当前事件的 Hook，并在 BLOCK 时停止。

        Run subscribed hooks in order and stop on BLOCK.
        """

        errors: list[HookError] = []
        for hook in self._hooks:
            if event.event_type not in hook.events:
                continue

            try:
                result = await hook.handle(event)
            except Exception as error:
                hook_error = HookError(
                    hook_name=hook.name,
                    event_type=event.event_type,
                    error_type=type(error).__name__,
                )
                errors.append(hook_error)
                logger.error(
                    "hook execution failed",
                    extra={
                        "hook_name": hook.name,
                        "hook_event": event.event_type.value,
                        "hook_error_type": hook_error.error_type,
                    },
                )
                if self.failure_mode is HookFailureMode.RAISE:
                    raise HookExecutionError(
                        f"hook execution failed: {hook.name}: {hook_error.error_type}"
                    ) from error
                continue

            active_result = result or HookResult()
            if active_result.decision is HookDecision.BLOCK:
                return HookDispatchResult(
                    decision=HookDecision.BLOCK,
                    reason=active_result.reason,
                    hook_name=hook.name,
                    errors=tuple(errors),
                )

        return HookDispatchResult(errors=tuple(errors))

    def dispatch_sync(self, event: HookEvent) -> HookDispatchResult:
        """从同步 Graph Node 执行异步 Hook。

        Run asynchronous hooks from a synchronous graph node.
        """

        coroutine: Coroutine[Any, Any, HookDispatchResult] = self.dispatch(event)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        coroutine.close()
        raise RuntimeError("use the asynchronous agent loop inside an active event loop")


class PermissionCheckHook:
    """把 Permission Pipeline 的结果桥接到 PreToolUse 阻断语义。

    Bridge permission-pipeline results into PreToolUse blocking semantics.
    """

    name = "permission_check"
    events = frozenset({HookEventType.PRE_TOOL_USE})

    async def handle(self, event: HookEvent) -> HookResult | None:
        if not isinstance(event, PreToolUse):
            return None
        if event.permission_granted:
            return None
        return HookResult(
            decision=HookDecision.BLOCK,
            reason=event.permission_result.reason,
        )


class ToolCallLoggingHook:
    """记录 Tool 名称、调用 ID 和结果状态，不记录参数或完整输出。

    Log tool name, call ID, and status without input or full output content.
    """

    name = "tool_call_logging"
    events = frozenset({HookEventType.PRE_TOOL_USE, HookEventType.POST_TOOL_USE})

    async def handle(self, event: HookEvent) -> HookResult | None:
        correlation = {
            "thread_id": event.state["thread_id"],
            "run_id": event.state.get("metadata", {}).get("run_id"),
        }
        if isinstance(event, PreToolUse):
            logger.info(
                "tool call started",
                extra={
                    **correlation,
                    "tool_name": event.tool_use.name,
                    "tool_use_id": event.tool_use.id,
                },
            )
        elif isinstance(event, PostToolUse):
            logger.info(
                "tool call finished",
                extra={
                    **correlation,
                    "tool_name": event.tool_use.name,
                    "tool_use_id": event.tool_use.id,
                    "tool_is_error": event.tool_result.is_error,
                },
            )
        return None


class LargeOutputWarningHook:
    """Tool 输出过大时记录告警，但不记录输出正文。

    Warn when tool output is large without logging its content.
    """

    name = "large_output_warning"
    events = frozenset({HookEventType.POST_TOOL_USE})

    def __init__(self, max_chars: int = DEFAULT_LARGE_OUTPUT_CHARS) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be greater than 0")
        self.max_chars = max_chars

    async def handle(self, event: HookEvent) -> HookResult | None:
        if not isinstance(event, PostToolUse):
            return None

        content = event.tool_result.content
        serialized = (
            content
            if isinstance(content, str)
            else json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        )
        if len(serialized) > self.max_chars:
            logger.warning(
                "tool output exceeds warning threshold",
                extra={
                    "tool_name": event.tool_use.name,
                    "tool_use_id": event.tool_use.id,
                    "tool_output_chars": len(serialized),
                },
            )
        return None


class StopMetricsHook:
    """按停止原因统计当前进程内的 Agent Turn 数量。

    Count agent turns by stop reason within the current process.
    """

    name = "stop_metrics"
    events = frozenset({HookEventType.STOP})

    def __init__(self) -> None:
        self.counts: Counter[AgentStopReason] = Counter()

    async def handle(self, event: HookEvent) -> HookResult | None:
        if isinstance(event, Stop):
            self.counts[event.reason] += 1
        return None


__all__ = [
    "DEFAULT_LARGE_OUTPUT_CHARS",
    "DuplicateHookError",
    "Hook",
    "HookDecision",
    "HookDispatchResult",
    "HookError",
    "HookEvent",
    "HookEventType",
    "HookExecutionError",
    "HookFailureMode",
    "HookRegistry",
    "HookResult",
    "LargeOutputWarningHook",
    "PermissionCheckHook",
    "PostToolUse",
    "PreToolUse",
    "Stop",
    "StopMetricsHook",
    "ToolCallLoggingHook",
    "UserPromptSubmit",
]
