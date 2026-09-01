"""M7 Subagent 隔离运行时测试。"""

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
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.subagent import SubagentRunner
from harness.messages import Message, MessageRole
from harness.model import ModelProvider
from services.checkpoint import create_in_memory_checkpointer


def _child_loop(response: str):
    return create_agent_loop(
        cast(ModelProvider, FakeModel(Message(role=MessageRole.ASSISTANT, content=response))),
        lambda: "isolated researcher",
        create_in_memory_checkpointer(),
    )


@pytest.mark.asyncio
async def test_subagent_runner_uses_new_thread_and_only_task_context() -> None:
    captured_context: SubagentContext | None = None

    def factory(definition: SubagentDefinition, context: SubagentContext) -> AgentLoop:
        nonlocal captured_context
        captured_context = context
        return _child_loop("evidence [S1]")

    runner = SubagentRunner(factory)
    result = await runner.run(
        SubagentTask(
            task_id="task-one",
            role="rag_researcher",
            objective="find the relevant interview record",
            context={"requested_search_mode": "rag"},
        ),
        SubagentContext(
            parent_thread_id="conversation-one",
            parent_run_id="run-one",
            user_id="user-one",
            search_mode="rag",
        ),
        SubagentDefinition(
            role="rag_researcher",
            system_prompt="research",
            allowed_tool_names=("document_search",),
        ),
    )

    assert result.status is SubagentStatus.SUCCEEDED
    assert result.child_thread_id == "conversation-one:subagent:task-one"
    assert result.summary == "evidence [S1]"
    assert captured_context is not None
    assert captured_context.user_id == "user-one"


@pytest.mark.asyncio
async def test_team_coordinator_enforces_depth_and_task_budget() -> None:
    calls: list[str] = []

    def factory(definition: SubagentDefinition, context: SubagentContext) -> AgentLoop:
        calls.append(definition.role)
        return _child_loop("done")

    definition = SubagentDefinition(
        role="rag_researcher",
        system_prompt="research",
    )
    coordinator = TeamCoordinator(
        SubagentRunner(factory),
        {definition.role: definition},
        DelegationBudget(max_tasks=1, max_depth=1),
    )
    task = SubagentTask(role=definition.role, objective="one")
    context = SubagentContext(parent_thread_id="thread")

    first = await coordinator.delegate(task, context)
    second = await coordinator.delegate(
        SubagentTask(role=definition.role, objective="two"),
        context,
    )
    too_deep = await coordinator.delegate(
        SubagentTask(role=definition.role, objective="three"),
        context.model_copy(update={"depth": 1}),
    )

    assert first.succeeded
    assert second.status is SubagentStatus.FAILED
    assert "task budget" in (second.error_reason or "")
    assert too_deep.status is SubagentStatus.FAILED
    assert "depth budget" in (too_deep.error_reason or "")
    assert calls == ["rag_researcher"]
