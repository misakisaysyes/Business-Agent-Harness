"""Conversation 生命周期和单会话 Run 控制。

Conversation lifecycle and per-conversation run control.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from harness.agent_loop import AgentLoop, get_permission_request
from harness.logging import AgentLog, new_trace_id
from harness.messages import Message, MessageRole
from harness.permissions import PermissionApproval, PermissionRequest
from harness.state import AgentState

log = AgentLog(__name__)


class ConversationStatus(StrEnum):
    """Conversation 当前是否可以接收新消息。

    Whether a conversation can currently accept a new message.
    """

    IDLE = "idle"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"


class ConversationError(RuntimeError):
    """Conversation 操作失败的基类。"""


class ConversationNotFoundError(ConversationError):
    """Conversation 不存在。"""


class ConversationForbiddenError(ConversationError):
    """Conversation 不属于当前用户。"""


class ConversationBusyError(ConversationError):
    """Conversation 已有活跃 Run。"""


class InvalidConversationInputError(ConversationError, ValueError):
    """Conversation 输入不符合最小约束。"""


class RunNotFoundError(ConversationError):
    """待恢复的 Run 不存在或不匹配。"""


class ConversationRunCancelledError(ConversationError):
    """正在执行的 Run 已被用户取消。"""


@dataclass(slots=True)
class ConversationRecord:
    """可持久化的 Conversation 元数据。

    Persistable conversation metadata.
    """

    conversation_id: str
    user_id: str
    status: ConversationStatus = ConversationStatus.IDLE
    active_run_id: str | None = None
    permission_request: PermissionRequest | None = None


class ConversationStore(Protocol):
    """Conversation 元数据存储的最小契约。

    Minimal storage contract for conversation metadata.
    """

    def create(self, record: ConversationRecord) -> None: ...

    def get(self, conversation_id: str) -> ConversationRecord | None: ...

    def list(self, user_id: str) -> tuple[ConversationRecord, ...]: ...

    def save(self, record: ConversationRecord) -> None: ...

    def delete(self, conversation_id: str) -> None: ...

    def recover_running(self) -> None: ...


class InMemoryConversationStore:
    """单元测试可使用的进程内 Conversation Store。"""

    def __init__(self) -> None:
        self._records: dict[str, ConversationRecord] = {}

    def create(self, record: ConversationRecord) -> None:
        self._records[record.conversation_id] = record

    def get(self, conversation_id: str) -> ConversationRecord | None:
        return self._records.get(conversation_id)

    def list(self, user_id: str) -> tuple[ConversationRecord, ...]:
        return tuple(record for record in self._records.values() if record.user_id == user_id)

    def save(self, record: ConversationRecord) -> None:
        self._records[record.conversation_id] = record

    def delete(self, conversation_id: str) -> None:
        del self._records[conversation_id]

    def recover_running(self) -> None:
        for record in self._records.values():
            if record.status is ConversationStatus.RUNNING:
                record.status = ConversationStatus.IDLE
                record.active_run_id = None
                record.permission_request = None


@dataclass(frozen=True, slots=True)
class ConversationRunResult:
    """一次消息处理或 Permission 恢复的结果。

    Result of one message execution or permission resume.
    """

    conversation_id: str
    run_id: str
    status: ConversationStatus
    messages: tuple[Message, ...]
    permission_request: PermissionRequest | None = None


class ConversationService:
    """管理 Conversation 所有权，并调用对应用户的 AgentLoop。

    Manage conversation ownership and invoke the owning user's agent loop.
    """

    def __init__(
        self,
        get_agent_loop: Callable[[str], AgentLoop],
        store: ConversationStore | None = None,
    ) -> None:
        self._get_agent_loop = get_agent_loop
        self._store = store or InMemoryConversationStore()
        self._active_tasks: dict[tuple[str, str], asyncio.Task[AgentState]] = {}
        # 进程在普通模型调用中退出时没有可恢复的 interrupt 输入；清除陈旧锁，
        # 但保留 WAITING_PERMISSION，使用户能用原 run_id 继续审批。
        # A crash during a normal model call has no resumable interrupt input; clear
        # that stale lock while retaining WAITING_PERMISSION for approval recovery.
        self._store.recover_running()

    def create(self, user_id: str) -> ConversationRecord:
        """为用户创建 Conversation，并确保其 Runtime 已就绪。

        Create a conversation for a user and ensure their runtime is ready.
        """

        self._get_agent_loop(user_id)
        conversation_id = uuid4().hex
        record = ConversationRecord(conversation_id=conversation_id, user_id=user_id)
        self._store.create(record)
        log.record("agent.conversation.created", conversation_id=conversation_id)
        return record

    def list(self, user_id: str) -> tuple[ConversationRecord, ...]:
        """返回当前用户的全部 Conversation。

        Return every conversation owned by the current user.
        """

        self._get_agent_loop(user_id)
        return self._store.list(user_id)

    def get(self, user_id: str, conversation_id: str) -> ConversationRecord:
        """返回当前用户拥有的 Conversation 详情。"""

        self._get_agent_loop(user_id)
        return self._owned_conversation(user_id, conversation_id)

    async def delete(self, user_id: str, conversation_id: str) -> None:
        """删除空闲 Conversation 及其 Checkpoint。

        Delete an idle conversation together with its checkpoints.
        """

        record = self._owned_conversation(user_id, conversation_id)
        if record.status is not ConversationStatus.IDLE:
            raise ConversationBusyError(
                f"cannot delete a conversation with an active run: {record.active_run_id}"
            )

        with (
            log.bind(conversation_id=conversation_id, thread_id=conversation_id),
            log.operation("agent.conversation.delete"),
        ):
            await self._get_agent_loop(user_id).adelete_thread(conversation_id)
            self._store.delete(conversation_id)

    async def send_message(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        required_tool: str | None = None,
    ) -> ConversationRunResult:
        """执行一条用户消息；同一 Conversation 有活跃 Run 时立即拒绝。

        Execute one user message and reject an already-active conversation.
        """

        if not content.strip():
            raise InvalidConversationInputError("message content must not be empty")

        agent_loop = self._get_agent_loop(user_id)
        if required_tool is not None and required_tool not in agent_loop.tool_names:
            raise InvalidConversationInputError(
                f"required tool is not available: {required_tool}"
            )

        record = self._owned_conversation(user_id, conversation_id)
        if record.status is not ConversationStatus.IDLE:
            raise ConversationBusyError(
                f"conversation already has an active run: {record.active_run_id}"
            )

        run_id = uuid4().hex
        record.status = ConversationStatus.RUNNING
        record.active_run_id = run_id
        record.permission_request = None
        self._store.save(record)

        current_trace_id = log.context_fields().get("trace_id")
        trace_id = (
            current_trace_id if isinstance(current_trace_id, str) else new_trace_id()
        )
        with (
            log.bind(
                trace_id=trace_id,
                conversation_id=conversation_id,
                thread_id=conversation_id,
                run_id=run_id,
            ),
            log.operation(
                "agent.run",
                message_count=1,
                required_tool=required_tool,
            ) as outcome,
        ):
            metadata: dict[str, JsonValue] = {
                "user_id": user_id,
                "run_id": run_id,
                "trace_id": trace_id,
            }
            if required_tool is not None:
                metadata["required_tool"] = required_tool

            state: AgentState = {
                "thread_id": conversation_id,
                "messages": [Message(role=MessageRole.USER, content=content)],
                "metadata": metadata,
            }

            task_key = (conversation_id, run_id)
            graph_task = asyncio.create_task(agent_loop.ainvoke(state))
            self._active_tasks[task_key] = graph_task
            try:
                result = await graph_task
            except asyncio.CancelledError as error:
                self._finish(record)
                self._store.save(record)
                raise ConversationRunCancelledError("run was cancelled by the user") from error
            except Exception:
                self._finish(record)
                self._store.save(record)
                raise
            finally:
                self._active_tasks.pop(task_key, None)
            completed = self._complete_or_wait(record, run_id, result)
            outcome["status"] = completed.status.value
            outcome["message_count"] = len(completed.messages)
            return completed

    async def cancel_run(
        self,
        user_id: str,
        conversation_id: str,
        run_id: str,
    ) -> ConversationRecord:
        """取消当前进程中正在执行的模型/Graph Task。

        Cancel a model/graph task currently running in this process.
        """

        record = self._owned_conversation(user_id, conversation_id)
        if record.status is not ConversationStatus.RUNNING or record.active_run_id != run_id:
            raise RunNotFoundError("run is not currently running")

        task = self._active_tasks.get((conversation_id, run_id))
        if task is None:
            raise RunNotFoundError("active run task is unavailable in this process")

        current_trace_id = log.context_fields().get("trace_id")
        trace_id = (
            current_trace_id if isinstance(current_trace_id, str) else new_trace_id()
        )
        with (
            log.bind(
                trace_id=trace_id,
                conversation_id=conversation_id,
                thread_id=conversation_id,
                run_id=run_id,
            ),
            log.operation("agent.run.cancel"),
        ):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._finish(record)
            self._store.save(record)
            return record

    async def resume_permission(
        self,
        user_id: str,
        conversation_id: str,
        run_id: str,
        approval: PermissionApproval | bool,
    ) -> ConversationRunResult:
        """使用审批结果恢复正在等待 Permission 的同一 Run。

        Resume the same run waiting for permission with an approval decision.
        """

        record = self._owned_conversation(user_id, conversation_id)
        if (
            record.status is not ConversationStatus.WAITING_PERMISSION
            or record.active_run_id != run_id
        ):
            raise RunNotFoundError("run is not waiting for permission")

        current_trace_id = log.context_fields().get("trace_id")
        trace_id = (
            current_trace_id if isinstance(current_trace_id, str) else new_trace_id()
        )
        with (
            log.bind(
                trace_id=trace_id,
                conversation_id=conversation_id,
                thread_id=conversation_id,
                run_id=run_id,
            ),
            log.operation("agent.run.resume") as outcome,
        ):
            record.status = ConversationStatus.RUNNING
            record.permission_request = None
            self._store.save(record)
            try:
                result = await self._get_agent_loop(user_id).aresume(
                    conversation_id,
                    approval,
                )
            except Exception:
                self._finish(record)
                self._store.save(record)
                raise
            completed = self._complete_or_wait(record, run_id, result)
            outcome["status"] = completed.status.value
            outcome["message_count"] = len(completed.messages)
            return completed

    def _owned_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ConversationRecord:
        record = self._store.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(f"unknown conversation: {conversation_id}")
        if record.user_id != user_id:
            raise ConversationForbiddenError("conversation belongs to another user")
        return record

    def _complete_or_wait(
        self,
        record: ConversationRecord,
        run_id: str,
        result: AgentState,
    ) -> ConversationRunResult:
        permission_request = get_permission_request(result)
        if permission_request is not None:
            record.status = ConversationStatus.WAITING_PERMISSION
            record.permission_request = permission_request
        else:
            self._finish(record)
        self._store.save(record)

        return ConversationRunResult(
            conversation_id=record.conversation_id,
            run_id=run_id,
            status=record.status,
            messages=tuple(result["messages"]),
            permission_request=permission_request,
        )

    @staticmethod
    def _finish(record: ConversationRecord) -> None:
        record.status = ConversationStatus.IDLE
        record.active_run_id = None
        record.permission_request = None


__all__ = [
    "ConversationBusyError",
    "ConversationForbiddenError",
    "ConversationNotFoundError",
    "ConversationRecord",
    "ConversationRunCancelledError",
    "ConversationRunResult",
    "ConversationService",
    "ConversationStatus",
    "ConversationStore",
    "InMemoryConversationStore",
    "InvalidConversationInputError",
    "RunNotFoundError",
]
