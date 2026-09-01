"""M8 Analyst、Reviewer 和 Lead 工具测试。"""

from typing import cast

import pytest
from tests.fakes import FakeModel

from business.knowledge_assistant.agent_teams.analyst import (
    ANALYST,
    AnalystFinding,
    AnalystOutput,
    build_analyst_definition,
)
from business.knowledge_assistant.agent_teams.lead import (
    DelegateAnalysisTool,
    RequestReviewTool,
)
from business.knowledge_assistant.agent_teams.reviewer import (
    REVIEWER,
    ReviewDecision,
    ReviewOutput,
    build_reviewer_definition,
)
from harness.agent_loop import AgentLoop, create_agent_loop
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.subagent import SubagentRunner
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.tool_use import ToolExecutionContext
from services.checkpoint import create_in_memory_checkpointer


def _loop(response: str) -> AgentLoop:
    return create_agent_loop(
        cast(
            ModelProvider,
            FakeModel(Message(role=MessageRole.ASSISTANT, content=response)),
        ),
        lambda: "role child",
        create_in_memory_checkpointer(),
    )


def _coordinator() -> TeamCoordinator:
    definitions = {
        ANALYST: build_analyst_definition({"calculator"}),
        REVIEWER: build_reviewer_definition(),
    }
    return TeamCoordinator(
        SubagentRunner(lambda definition, context: _loop(f"{definition.role} done")),
        definitions,
    )


def test_m8_role_definitions_have_separate_tool_boundaries() -> None:
    analyst = build_analyst_definition({"calculator", "document_search"})
    reviewer = build_reviewer_definition()

    assert analyst.role == ANALYST
    assert analyst.allowed_tool_names == ("calculator",)
    assert reviewer.role == REVIEWER
    assert reviewer.allowed_tool_names == ()


def test_m8_outputs_are_strict_structured_contracts() -> None:
    finding = AnalystFinding(statement="A is supported", citation_ids=("S1",))
    output = AnalystOutput(findings=(finding,), conclusions=("A",))
    review = ReviewOutput(
        decision=ReviewDecision.APPROVED,
        checked_citation_ids=("S1",),
    )

    assert output.findings[0].citation_ids == ("S1",)
    assert review.decision is ReviewDecision.APPROVED


@pytest.mark.asyncio
async def test_lead_can_delegate_analysis_and_review_with_scoped_context() -> None:
    coordinator = _coordinator()
    analysis_tool = DelegateAnalysisTool(coordinator)
    review_tool = RequestReviewTool(coordinator)
    context = ToolExecutionContext(
        thread_id="parent-thread",
        metadata={"search_mode": "rag", "user_id": "user-one"},
    )

    analysis = await analysis_tool.ainvoke_with_context(
        ToolUse(
            id="analysis-call",
            name="delegate_analysis",
            input={
                "objective": "compare the supplied records",
                "evidence": ["record A"],
                "citation_ids": ["S1"],
            },
        ),
        context,
    )
    review = await review_tool.ainvoke_with_context(
        ToolUse(
            id="review-call",
            name="request_review",
            input={
                "candidate": {"answer": "A"},
                "evidence": ["record A"],
                "citation_ids": ["S1"],
            },
        ),
        context,
    )

    assert not analysis.is_error
    assert not review.is_error
