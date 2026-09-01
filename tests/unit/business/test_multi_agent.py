"""Knowledge Assistant M7 Lead/Researcher 委派测试。"""

from typing import cast

import pytest
from pydantic import JsonValue
from tests.fakes import FakeModel

from business.knowledge_assistant.agent_teams.lead import DelegateResearchTool
from business.knowledge_assistant.agent_teams.researcher import (
    RAG_RESEARCHER,
    WEB_RESEARCHER,
    build_researcher_definitions,
)
from harness.agent_loop import AgentLoop, create_agent_loop
from harness.capabilities.agent_teams.contracts import SubagentContext, SubagentDefinition
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.subagent import SubagentRunner
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.tool_use import ToolExecutionContext
from services.checkpoint import create_in_memory_checkpointer


def _loop(response: str):
    return create_agent_loop(
        cast(ModelProvider, FakeModel(Message(role=MessageRole.ASSISTANT, content=response))),
        lambda: "researcher",
        create_in_memory_checkpointer(),
    )


@pytest.mark.asyncio
async def test_delegate_research_preserves_role_and_search_mode() -> None:
    definitions = build_researcher_definitions({"document_search", "mcp__search__web_search"})
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    def factory(definition: SubagentDefinition, context: SubagentContext) -> AgentLoop:
        seen.append((definition.role, context.search_mode, definition.allowed_tool_names))
        return _loop("private evidence [S1]")

    coordinator = TeamCoordinator(SubagentRunner(factory), definitions)
    tool = DelegateResearchTool(coordinator)
    result = await tool.ainvoke_with_context(
        ToolUse(
            id="delegate-one",
            name="delegate_research",
            input={"objective": "find private interview evidence", "research_kind": "rag"},
        ),
        ToolExecutionContext(
            thread_id="parent-thread",
            metadata={"search_mode": "rag", "user_id": "user-one"},
        ),
    )

    assert not result.is_error
    content = cast(dict[str, JsonValue], result.content)
    assert content["status"] == "succeeded"
    assert seen == [(RAG_RESEARCHER, "rag", ("document_search",))]


@pytest.mark.asyncio
async def test_delegate_research_rejects_hybrid_as_one_unscoped_task() -> None:
    definitions = build_researcher_definitions({"document_search", "mcp__search__web_search"})
    coordinator = TeamCoordinator(
        SubagentRunner(lambda definition, context: _loop("unused")),
        definitions,
    )
    tool = DelegateResearchTool(coordinator)
    result = await tool.ainvoke_with_context(
        ToolUse(
            id="delegate-hybrid",
            name="delegate_research",
            input={"objective": "compare my records with the latest public news"},
        ),
        ToolExecutionContext(
            thread_id="parent-thread",
            metadata={"search_mode": "hybrid"},
        ),
    )

    assert result.is_error
    content = cast(dict[str, JsonValue], result.content)
    assert "split" in str(content["message"])
    assert WEB_RESEARCHER in definitions
