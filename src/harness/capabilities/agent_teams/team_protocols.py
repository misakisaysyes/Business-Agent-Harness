"""Agent Teams 的结构化请求—响应协议。

Structured request-response protocols for Agent Teams.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TeamMessageKind(StrEnum):
    """Team 内允许传递的消息类型。"""

    TASK_REQUEST = "TASK_REQUEST"
    TASK_ACCEPTED = "TASK_ACCEPTED"
    TASK_REJECTED = "TASK_REJECTED"
    TASK_RESULT = "TASK_RESULT"
    TASK_FAILED = "TASK_FAILED"
    RESULT_ACK = "RESULT_ACK"
    REVIEW_REQUEST = "REVIEW_REQUEST"
    REVIEW_RESULT = "REVIEW_RESULT"
    REVISION_REQUEST = "REVISION_REQUEST"
    PERMISSION_REQUEST = "PERMISSION_REQUEST"
    PERMISSION_RESPONSE = "PERMISSION_RESPONSE"
    RETRY = "RETRY"
    DEGRADED = "DEGRADED"
    SHUTDOWN = "SHUTDOWN"
    SHUTDOWN_ACK = "SHUTDOWN_ACK"


class TeamProtocolError(ValueError):
    """结构化 Team 消息不符合协议。"""


class TeamMessage(BaseModel):
    """Team MessageBus 传输的不可变消息。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}", min_length=1)
    team_run_id: str = Field(min_length=1)
    parent_run_id: str | None = None
    task_id: str = Field(min_length=1)
    parent_task_id: str | None = None
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    kind: TeamMessageKind
    correlation_id: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# 便于调用方按计划中的泛化命名导入。
MessageKind: Final = TeamMessageKind
ProtocolMessage: Final = TeamMessage


def validate_team_message(message: TeamMessage) -> TeamMessage:
    """校验发送者、接收者和协议消息的基本方向。"""

    if message.sender == message.recipient:
        raise TeamProtocolError("sender and recipient must be different")

    sender = message.sender.lower()
    recipient = message.recipient.lower()
    teammate_sender = sender not in {"lead", "runtime", "user"}
    teammate_recipient = recipient not in {"lead", "runtime", "user"}

    sender_rules = {
        TeamMessageKind.TASK_REQUEST: sender == "lead",
        TeamMessageKind.TASK_ACCEPTED: sender in {"runtime", "lead"},
        TeamMessageKind.TASK_REJECTED: sender in {"runtime", "lead"},
        TeamMessageKind.REVIEW_REQUEST: sender == "lead",
        TeamMessageKind.TASK_RESULT: teammate_sender,
        TeamMessageKind.TASK_FAILED: teammate_sender or sender == "runtime",
        TeamMessageKind.RESULT_ACK: sender == "lead",
        TeamMessageKind.REVIEW_RESULT: sender == "reviewer",
        TeamMessageKind.REVISION_REQUEST: sender == "reviewer",
        TeamMessageKind.PERMISSION_REQUEST: teammate_sender,
        TeamMessageKind.PERMISSION_RESPONSE: sender in {"lead", "runtime", "user"},
        TeamMessageKind.RETRY: sender == "runtime",
        TeamMessageKind.DEGRADED: sender in {"lead", "runtime"},
        TeamMessageKind.SHUTDOWN: sender in {"lead", "runtime"},
    }
    if message.kind in sender_rules and not sender_rules[message.kind]:
        raise TeamProtocolError(
            f"invalid sender for {message.kind.value}: {message.sender}"
        )

    recipient_rules = {
        TeamMessageKind.TASK_RESULT: recipient == "lead",
        TeamMessageKind.TASK_FAILED: recipient == "lead",
        TeamMessageKind.RESULT_ACK: teammate_recipient,
        TeamMessageKind.REVIEW_RESULT: recipient == "lead",
        TeamMessageKind.REVISION_REQUEST: recipient == "lead",
        TeamMessageKind.PERMISSION_REQUEST: recipient == "lead",
        TeamMessageKind.PERMISSION_RESPONSE: recipient in {"lead", "runtime"}
        or recipient == "user",
        TeamMessageKind.SHUTDOWN_ACK: recipient in {"lead", "runtime"},
    }
    if message.kind in recipient_rules and not recipient_rules[message.kind]:
        raise TeamProtocolError(
            f"invalid recipient for {message.kind.value}: {message.recipient}"
        )
    if message.kind is TeamMessageKind.TASK_REQUEST and not teammate_recipient:
        raise TeamProtocolError("TASK_REQUEST must target a teammate")
    return message


class TeamProtocolState:
    """校验单个 Team Run 内任务的基本状态转换和消息去重。"""

    def __init__(self) -> None:
        self._task_status: dict[tuple[str, str], str] = {}
        self._seen_message_ids: set[tuple[str, str]] = set()

    def accept(self, message: TeamMessage) -> bool:
        """接受一条消息；重复消息返回 False，非法转换抛出错误。"""

        validate_team_message(message)
        message_key = (message.team_run_id, message.message_id)
        if message_key in self._seen_message_ids:
            return False

        task_key = (message.team_run_id, message.task_id)
        current = self._task_status.get(task_key, "pending")
        transitions = {
            TeamMessageKind.TASK_REQUEST: {"pending", "failed", "rejected"},
            TeamMessageKind.TASK_ACCEPTED: {"pending"},
            TeamMessageKind.TASK_REJECTED: {"pending"},
            TeamMessageKind.TASK_RESULT: {"running", "pending"},
            TeamMessageKind.TASK_FAILED: {"running", "pending"},
            TeamMessageKind.RETRY: {"failed", "timeout"},
            TeamMessageKind.REVISION_REQUEST: {"succeeded"},
        }
        allowed = transitions.get(message.kind)
        if allowed is not None and current not in allowed:
            raise TeamProtocolError(
                f"invalid task transition for {message.task_id}: "
                f"{current} -> {message.kind.value}"
            )

        self._seen_message_ids.add(message_key)

        status_by_kind = {
            TeamMessageKind.TASK_ACCEPTED: "running",
            TeamMessageKind.TASK_REJECTED: "rejected",
            TeamMessageKind.TASK_RESULT: "succeeded",
            TeamMessageKind.TASK_FAILED: "failed",
            TeamMessageKind.RETRY: "pending",
            TeamMessageKind.REVISION_REQUEST: "pending",
        }
        if status := status_by_kind.get(message.kind):
            self._task_status[task_key] = status
        return True

    def status(self, team_run_id: str, task_id: str) -> str:
        """读取任务状态；未出现过的任务视为 pending。"""

        return self._task_status.get((team_run_id, task_id), "pending")


def make_team_message(
    *,
    team_run_id: str,
    task_id: str,
    sender: str,
    recipient: str,
    kind: TeamMessageKind,
    payload: dict[str, JsonValue] | None = None,
    parent_run_id: str | None = None,
    parent_task_id: str | None = None,
    correlation_id: str | None = None,
) -> TeamMessage:
    """创建并校验一条协议消息。"""

    message = TeamMessage(
        team_run_id=team_run_id,
        parent_run_id=parent_run_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        sender=sender,
        recipient=recipient,
        kind=kind,
        correlation_id=correlation_id or task_id,
        payload=payload or {},
    )
    return validate_team_message(message)


__all__ = [
    "MessageKind",
    "ProtocolMessage",
    "TeamMessage",
    "TeamMessageKind",
    "TeamProtocolError",
    "TeamProtocolState",
    "make_team_message",
    "validate_team_message",
]
