"""M8 MessageBus、协议和 Team 调度测试。"""

from typing import cast

import pytest
from tests.fakes import FakeModel

from harness.agent_loop import AgentLoop, create_agent_loop
from harness.capabilities.agent_teams.contracts import (
    DelegationBudget,
    SubagentContext,
    SubagentDefinition,
    SubagentStatus,
    SubagentTask,
)
from harness.capabilities.agent_teams.message_bus import (
    InMemoryMessageBus,
    MessageBusClosedError,
)
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.agent_teams.team_protocols import (
    TeamMessageKind,
    TeamProtocolState,
    make_team_message,
)
from harness.capabilities.subagent import SubagentRunner
from harness.messages import Message, MessageRole
from harness.model import ModelProvider
from services.checkpoint import create_in_memory_checkpointer


def _loop(response: str) -> AgentLoop:
    return create_agent_loop(
        cast(
            ModelProvider,
            FakeModel(Message(role=MessageRole.ASSISTANT, content=response)),
        ),
        lambda: "m8 child",
        create_in_memory_checkpointer(),
    )


@pytest.mark.asyncio
async def test_in_memory_message_bus_deduplicates_and_closes() -> None:
    bus = InMemoryMessageBus()
    received: list[str] = []

    async def handler(message) -> None:  # type: ignore[no-untyped-def]
        received.append(message.message_id)

    bus.subscribe("rag_researcher", handler)
    message = make_team_message(
        team_run_id="team-one",
        task_id="task-one",
        sender="lead",
        recipient="rag_researcher",
        kind=TeamMessageKind.TASK_REQUEST,
    )

    assert await bus.send(message)
    assert not await bus.send(message)
    assert received == [message.message_id]
    assert bus.messages("team-one") == (message,)

    await bus.close("team-one")
    with pytest.raises(MessageBusClosedError):
        await bus.send(message.model_copy(update={"message_id": "message-two"}))


def test_team_protocol_state_validates_task_lifecycle() -> None:
    state = TeamProtocolState()
    request = make_team_message(
        team_run_id="team-one",
        task_id="task-one",
        sender="lead",
        recipient="analyst",
        kind=TeamMessageKind.TASK_REQUEST,
    )
    accepted = make_team_message(
        team_run_id="team-one",
        task_id="task-one",
        sender="runtime",
        recipient="lead",
        kind=TeamMessageKind.TASK_ACCEPTED,
    )
    result = make_team_message(
        team_run_id="team-one",
        task_id="task-one",
        sender="analyst",
        recipient="lead",
        kind=TeamMessageKind.TASK_RESULT,
    )

    assert state.accept(request)
    assert state.accept(accepted)
    assert state.accept(result)
    assert state.status("team-one", "task-one") == "succeeded"
    assert not state.accept(result)


@pytest.mark.asyncio
async def test_team_coordinator_retries_only_failed_subtask() -> None:
    factory_calls = 0

    def factory(definition: SubagentDefinition, context: SubagentContext) -> AgentLoop:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("transient child failure")
        return _loop("recovered")

    definition = SubagentDefinition(role="analyst", system_prompt="analyze")
    coordinator = TeamCoordinator(
        SubagentRunner(factory),
        {definition.role: definition},
        DelegationBudget(
            max_retries=1,
            retry_base_delay_seconds=0,
            retry_jitter_seconds=0,
        ),
    )
    result = await coordinator.delegate(
        SubagentTask(role="analyst", objective="compare evidence"),
        SubagentContext(parent_thread_id="parent", team_run_id="team-one"),
    )

    assert result.status is SubagentStatus.SUCCEEDED
    assert result.attempt == 2
    assert result.attempts == 2
    assert factory_calls == 2
    events = coordinator.message_bus.messages("team-one")
    assert [event.kind for event in events] == [
        TeamMessageKind.TASK_REQUEST,
        TeamMessageKind.TASK_ACCEPTED,
        TeamMessageKind.TASK_FAILED,
        TeamMessageKind.RETRY,
        TeamMessageKind.TASK_REQUEST,
        TeamMessageKind.TASK_ACCEPTED,
        TeamMessageKind.TASK_RESULT,
        TeamMessageKind.RESULT_ACK,
    ]


@pytest.mark.asyncio
async def test_team_coordinator_delegate_many_returns_input_order() -> None:
    calls: list[str] = []

    def factory(definition: SubagentDefinition, context: SubagentContext) -> AgentLoop:
        calls.append(context.parent_thread_id)
        return _loop(definition.role)

    definitions = {
        role: SubagentDefinition(role=role, system_prompt=role)
        for role in ("analyst", "reviewer")
    }
    coordinator = TeamCoordinator(SubagentRunner(factory), definitions)
    results = await coordinator.delegate_many(
        (
            (
                SubagentTask(role="reviewer", objective="review"),
                SubagentContext(parent_thread_id="review"),
            ),
            (
                SubagentTask(role="analyst", objective="analyze"),
                SubagentContext(parent_thread_id="analysis"),
            ),
        )
    )

    assert [result.role for result in results] == ["reviewer", "analyst"]
    assert calls == ["review", "analysis"]


@pytest.mark.asyncio
async def test_review_rounds_are_bounded_to_initial_plus_two_revisions() -> None:
    definition = SubagentDefinition(role="reviewer", system_prompt="review")
    coordinator = TeamCoordinator(
        SubagentRunner(lambda *_: _loop("ok")),
        {"reviewer": definition},
    )
    context = SubagentContext(parent_thread_id="parent", team_run_id="team-review")

    assert await coordinator.reserve_review_round(context) == ("team-review", 1)
    assert await coordinator.reserve_review_round(context) == ("team-review", 2)
    assert await coordinator.reserve_review_round(context) == ("team-review", 3)
    assert await coordinator.reserve_review_round(context) is None
