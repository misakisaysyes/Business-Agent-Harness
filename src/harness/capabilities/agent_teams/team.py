"""Lead 调度、角色注册和任务委派。

Lead coordination, role registration, bounded delegation, and in-process
Team protocol events.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import cast
from uuid import uuid4

import structlog
from pydantic import JsonValue

from harness.capabilities.agent_teams.contracts import (
    DelegationBudget,
    SubagentContext,
    SubagentDefinition,
    SubagentResult,
    SubagentStatus,
    SubagentTask,
)
from harness.capabilities.agent_teams.message_bus import (
    InMemoryMessageBus,
    MessageBus,
    MessageBusClosedError,
)
from harness.capabilities.agent_teams.team_protocols import (
    TeamMessageKind,
    TeamProtocolState,
    make_team_message,
)
from harness.capabilities.subagent import SubagentRunner

logger = structlog.get_logger(__name__)
RetrySleeper = Callable[[float], Awaitable[None]]


class TeamCoordinator:
    """在进程内提供有界的 Agent-as-a-Tool 委派。

    ``delegate`` 是单任务边界；``delegate_many`` 用于无依赖任务的并行
    执行。每次重试只重新运行当前任务，已完成的兄弟任务不会被重跑。
    """

    def __init__(
        self,
        runner: SubagentRunner,
        definitions: Mapping[str, SubagentDefinition],
        budget: DelegationBudget | None = None,
        message_bus: MessageBus | None = None,
        retry_sleeper: RetrySleeper | None = None,
    ) -> None:
        self.runner = runner
        self.definitions = dict(definitions)
        self.budget = budget or DelegationBudget()
        self.message_bus = message_bus or InMemoryMessageBus()
        self._retry_sleeper = retry_sleeper or asyncio.sleep
        self._team_task_counts: dict[str, int] = {}
        self._team_ids: dict[str, str] = {}
        self._review_counts: dict[str, int] = {}
        self.protocol_state = TeamProtocolState()
        self._lock = asyncio.Lock()

    @property
    def available_roles(self) -> tuple[str, ...]:
        """返回已注册角色。"""

        return tuple(sorted(self.definitions))

    async def delegate(
        self,
        task: SubagentTask,
        context: SubagentContext,
    ) -> SubagentResult:
        """校验深度、预算和角色后执行一个可重试子任务。"""

        definition = self.definitions.get(task.role)
        if definition is None:
            return self._failure(task, f"unknown subagent role: {task.role}")
        if context.depth >= self.budget.max_depth:
            return self._failure(
                task,
                "subagent delegation depth budget exceeded",
                definition,
            )

        requested_tools = task.allowed_tool_names or definition.allowed_tool_names
        if not set(requested_tools).issubset(definition.allowed_tool_names):
            return SubagentResult(
                task_id=task.task_id,
                role=task.role,
                status=SubagentStatus.REJECTED,
                error_reason="task tool allowlist exceeds the role allowlist",
                tool_names=definition.allowed_tool_names,
            )

        team_run_id = await self._resolve_team_run_id(context)
        context = context.model_copy(
            update={
                "team_run_id": team_run_id,
                "allowed_tool_names": requested_tools,
            }
        )
        async with self._lock:
            task_count = self._team_task_counts.get(team_run_id, 0)
            if task_count >= self.budget.max_tasks:
                return self._failure(task, "subagent task budget exceeded", definition)
            self._team_task_counts[team_run_id] = task_count + 1

        max_attempts = self.budget.max_retries + 1
        last_result: SubagentResult | None = None
        for attempt in range(1, max_attempts + 1):
            await self._emit(
                task,
                context,
                TeamMessageKind.TASK_REQUEST,
                {
                    "objective": task.objective,
                    "role": task.role,
                    "attempt": attempt,
                },
            )
            await self._emit(
                task,
                context,
                TeamMessageKind.TASK_ACCEPTED,
                {"attempt": attempt},
                sender="runtime",
                recipient="lead",
            )
            logger.info(
                "subagent_task_started",
                team_run_id=team_run_id,
                parent_thread_id=context.parent_thread_id,
                parent_task_id=task.parent_task_id,
                task_id=task.task_id,
                role=task.role,
                attempt=attempt,
                allowed_tool_names=",".join(requested_tools),
            )
            result = await self._run_once(task, context, definition)
            result = result.model_copy(
                update={
                    "attempt": attempt,
                    "attempts": attempt,
                    "last_error": result.error_reason,
                }
            )
            last_result = result
            if result.succeeded:
                await self._emit(
                    task,
                    context,
                    TeamMessageKind.TASK_RESULT,
                    result.model_dump(mode="json"),
                )
                await self._emit(
                    task,
                    context,
                    TeamMessageKind.RESULT_ACK,
                    {"attempt": attempt},
                    sender="lead",
                    recipient=task.role,
                )
                logger.info(
                    "subagent_task_finished",
                    team_run_id=team_run_id,
                    parent_task_id=task.parent_task_id,
                    task_id=task.task_id,
                    role=task.role,
                    attempt=attempt,
                    status=result.status.value,
                )
                return result

            if attempt >= max_attempts or not self._is_retryable(result):
                await self._emit(
                    task,
                    context,
                    TeamMessageKind.TASK_FAILED,
                    result.model_dump(mode="json"),
                )
                logger.info(
                    "subagent_task_finished",
                    team_run_id=team_run_id,
                    parent_task_id=task.parent_task_id,
                    task_id=task.task_id,
                    role=task.role,
                    attempt=attempt,
                    status=result.status.value,
                    last_error=result.error_reason,
                )
                return result

            delay = self._backoff_delay(attempt)
            await self._emit(
                task,
                context,
                TeamMessageKind.TASK_FAILED,
                result.model_dump(mode="json"),
            )
            await self._emit(
                task,
                context,
                TeamMessageKind.RETRY,
                {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "delay_seconds": delay,
                    "error": result.error_reason,
                },
            )
            logger.warning(
                "subagent_task_retry",
                team_run_id=team_run_id,
                task_id=task.task_id,
                role=task.role,
                attempt=attempt,
                delay_seconds=delay,
                last_error=result.error_reason,
            )
            await self._retry_sleeper(delay)

        return last_result or self._failure(
            task,
            "subagent execution produced no result",
            definition,
        )

    async def delegate_many(
        self,
        tasks: Sequence[tuple[SubagentTask, SubagentContext]],
    ) -> tuple[SubagentResult, ...]:
        """并行执行一组无依赖任务，并按输入顺序返回结果。"""

        return tuple(
            await asyncio.gather(
                *(self.delegate(task, context) for task, context in tasks)
            )
        )

    async def close(self, team_run_id: str) -> None:
        """关闭一个 Team Run，拒绝后续消息并等待已开始的处理。"""

        await self.message_bus.close(team_run_id)

    async def reserve_review_round(
        self,
        context: SubagentContext,
        *,
        max_rounds: int = 3,
    ) -> tuple[str, int] | None:
        """为一次审核预留轮次；默认允许初审加两次有限修订。"""

        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        team_run_id = await self._resolve_team_run_id(context)
        async with self._lock:
            current = self._review_counts.get(team_run_id, 0)
            if current >= max_rounds:
                return None
            round_number = current + 1
            self._review_counts[team_run_id] = round_number
            return team_run_id, round_number

    async def emit_protocol(
        self,
        task: SubagentTask,
        context: SubagentContext,
        kind: TeamMessageKind,
        payload: dict[str, object],
        *,
        sender: str | None = None,
        recipient: str | None = None,
    ) -> None:
        """向业务角色暴露受协议状态机保护的事件出口。"""

        await self._emit(
            task,
            context,
            kind,
            payload,
            sender=sender,
            recipient=recipient,
        )

    async def _resolve_team_run_id(self, context: SubagentContext) -> str:
        if context.team_run_id:
            return context.team_run_id
        scope = context.parent_run_id or context.parent_thread_id
        async with self._lock:
            team_run_id = self._team_ids.get(scope)
            if team_run_id is None:
                team_run_id = f"team_{uuid4().hex}"
                self._team_ids[scope] = team_run_id
            return team_run_id

    async def _run_once(
        self,
        task: SubagentTask,
        context: SubagentContext,
        definition: SubagentDefinition,
    ) -> SubagentResult:
        try:
            if self.budget.max_duration_seconds is None:
                return await self.runner.run(task, context, definition)
            async with asyncio.timeout(self.budget.max_duration_seconds):
                return await self.runner.run(task, context, definition)
        except TimeoutError:
            return SubagentResult(
                task_id=task.task_id,
                role=task.role,
                status=SubagentStatus.TIMEOUT,
                error_reason="subagent task timed out",
                tool_names=definition.allowed_tool_names,
            )

    async def _emit(
        self,
        task: SubagentTask,
        context: SubagentContext,
        kind: TeamMessageKind,
        payload: dict[str, object],
        *,
        sender: str | None = None,
        recipient: str | None = None,
    ) -> None:
        team_run_id = context.team_run_id
        if not isinstance(team_run_id, str) or not team_run_id:
            return
        message_sender = sender or (
            "lead"
            if kind is TeamMessageKind.TASK_REQUEST
            else "runtime"
            if kind is TeamMessageKind.RETRY
            else task.role
        )
        message_recipient = recipient or (
            task.role
            if kind in {TeamMessageKind.TASK_REQUEST, TeamMessageKind.RETRY}
            else "lead"
        )
        safe_payload: dict[str, JsonValue] = {
            key: cast(JsonValue, value)
            for key, value in payload.items()
            if _is_json_value(value)
        }
        message = make_team_message(
            team_run_id=team_run_id,
            parent_run_id=context.parent_run_id,
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            sender=message_sender,
            recipient=message_recipient,
            kind=kind,
            payload=safe_payload,
            correlation_id=task.task_id,
        )
        try:
            self.protocol_state.accept(message)
            await self.message_bus.send(message)
        except MessageBusClosedError:
            logger.warning("team_message_dropped_after_close", task_id=task.task_id)
        except ValueError:
            logger.warning("team_message_rejected", task_id=task.task_id)

    def _backoff_delay(self, attempt: int) -> float:
        base = max(0.0, self.budget.retry_base_delay_seconds)
        maximum = max(base, self.budget.retry_max_delay_seconds)
        delay = min(maximum, base * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0.0, self.budget.retry_jitter_seconds)
        return round(
            min(maximum + self.budget.retry_jitter_seconds, delay + jitter),
            3,
        )

    @staticmethod
    def _is_retryable(result: SubagentResult) -> bool:
        if result.status not in {SubagentStatus.FAILED, SubagentStatus.TIMEOUT}:
            return False
        reason = (result.error_reason or "").lower()
        non_retryable_markers = (
            "permission",
            "unknown",
            "budget",
            "allowlist",
            "no configured tools",
            "invalid",
        )
        return not any(marker in reason for marker in non_retryable_markers)

    @staticmethod
    def _failure(
        task: SubagentTask,
        reason: str,
        definition: SubagentDefinition | None = None,
    ) -> SubagentResult:
        return SubagentResult(
            task_id=task.task_id,
            role=task.role,
            status=SubagentStatus.FAILED,
            error_reason=reason,
            tool_names=definition.allowed_tool_names if definition is not None else (),
        )


def _is_json_value(value: object) -> bool:
    """Restrict protocol payloads to JSON-compatible scalar/container values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        items = cast(list[object], value)
        return all(_is_json_value(item) for item in items)
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in items.items()
        )
    return False


__all__ = ["TeamCoordinator"]
