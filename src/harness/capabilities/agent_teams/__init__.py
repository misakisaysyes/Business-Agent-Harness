"""可复用的 Multi-Agent 能力。

Reusable Multi-Agent capability.
"""

from harness.capabilities.agent_teams.contracts import (
    AgentRole,
    DelegationBudget,
    SubagentContext,
    SubagentDefinition,
    TaskStatus,
    TeamRun,
    TeamTask,
    TeamTaskResult,
)
from harness.capabilities.agent_teams.message_bus import (
    InMemoryMessageBus,
    MessageBus,
    MessageBusClosedError,
    MessageBusError,
)
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.agent_teams.team_protocols import (
    TeamMessage,
    TeamMessageKind,
    TeamProtocolError,
    TeamProtocolState,
    make_team_message,
    validate_team_message,
)

__all__ = [
    "AgentRole",
    "DelegationBudget",
    "InMemoryMessageBus",
    "MessageBus",
    "MessageBusClosedError",
    "MessageBusError",
    "SubagentContext",
    "SubagentDefinition",
    "TaskStatus",
    "TeamCoordinator",
    "TeamRun",
    "TeamMessage",
    "TeamMessageKind",
    "TeamProtocolError",
    "TeamProtocolState",
    "TeamTask",
    "TeamTaskResult",
    "make_team_message",
    "validate_team_message",
]
