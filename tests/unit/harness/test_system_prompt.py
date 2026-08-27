"""System Prompt 稳定组装测试。

Stable system-prompt assembly tests.
"""

from harness.context import ContextFragment, ContextManager
from harness.messages import Message, MessageRole
from harness.state import AgentState
from harness.system_prompt import SystemPromptBuilder
from harness.tool_use import ToolDefinition


class RuntimeContextProvider:
    """返回固定 Runtime Context 的测试 Provider。"""

    name = "runtime"

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                key="task",
                title="Current Task",
                content=state["messages"][-1].content,
                priority=100,
            ),
        )


def test_system_prompt_sections_have_a_stable_order() -> None:
    """基础、工具、业务、Skill、Memory、Context 必须按固定顺序出现。"""

    state: AgentState = {
        "thread_id": "prompt-thread",
        "messages": [Message(role=MessageRole.USER, content="compare two documents")],
    }
    tool = ToolDefinition(
        name="document_search",
        description="Search authorized documents.",
        parameters={"type": "object", "secret": "must-not-be-rendered"},
    )
    builder = SystemPromptBuilder(
        lambda: "Knowledge Assistant business rules.",
        ContextManager((RuntimeContextProvider(),)),
        skill_summaries=("Summarize documents",),
        memory_entries=("User prefers short answers",),
    )

    prompt = builder.build(state, (tool,))

    headings = (
        "# Core Instructions",
        "# Available Tools",
        "# Business Instructions",
        "# Skill Summaries",
        "# Memory",
        "# Runtime Context",
    )
    assert [prompt.index(heading) for heading in headings] == sorted(
        prompt.index(heading) for heading in headings
    )
    assert "- document_search: Search authorized documents." in prompt
    assert "Knowledge Assistant business rules." in prompt
    assert "## Current Task\ncompare two documents" in prompt
    assert "current explicit request, follow the current request" in prompt
    assert "must-not-be-rendered" not in prompt
