"""AgentProfile 通用装配契约测试。

Tests for the shared AgentProfile composition contract.
"""

from typing import cast

import pytest
from pydantic import ValidationError

from harness.context import ContextFragment, ContextProvider
from harness.messages import ToolResult, ToolUse
from harness.permissions import PermissionDecision, PermissionRule
from harness.profile import AgentProfile, Capability, ModelConfigRef
from harness.state import AgentState
from harness.tool_use import Tool, ToolInput


class FakeToolInput(ToolInput):
    """FakeTool 的空参数 Schema。

    Empty input schema for FakeTool.
    """


class FakeTool:
    """用于 Profile 装配验证的测试 Tool。

    Test tool used to validate profile composition.
    """

    name = "fake_tool"
    description = "A fake tool."
    input_schema = FakeToolInput
    concurrency_group = None

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """返回固定 ToolResult。

        Return a fixed ToolResult.
        """

        return ToolResult(tool_use_id=tool_use.id, content="ok")


class FakePermissionRule:
    """用于 Profile 装配验证的测试 Permission Rule。

    Test permission rule used to validate profile composition.
    """

    name = "allow_fake_tool"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionDecision | None:
        """始终允许测试 ToolUse。

        Always allow the test ToolUse.
        """

        return PermissionDecision.ALLOW


class FakeContextProvider:
    """用于 Profile 装配验证的测试 Context Provider。

    Test context provider used to validate profile composition.
    """

    name = "fake_context"

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        """返回固定测试上下文。

        Return fixed test context.
        """

        return (
            ContextFragment(
                key="test",
                title="Test Context",
                content="context",
            ),
        )


def system_prompt() -> str:
    """返回固定测试 System Prompt。

    Return a fixed test system prompt.
    """

    return "You are a test agent."


def test_profile_requires_name_model_and_system_prompt() -> None:
    """缺少核心装配项时 Profile 必须校验失败。

    A profile must fail validation when core composition fields are missing.
    """

    with pytest.raises(ValidationError) as exc_info:
        AgentProfile.model_validate({})

    error_locations = {error["loc"] for error in exc_info.value.errors()}

    assert ("name",) in error_locations
    assert ("model",) in error_locations
    assert ("system_prompt",) in error_locations


def test_profile_composes_only_shared_contracts() -> None:
    """Profile 应组合 Model、Prompt、Tool、Permission、Context 和 Capability。

    A profile should compose model, prompt, tool, permission, context, and capability contracts.
    """

    tool = FakeTool()
    permission_rule = FakePermissionRule()
    context_provider = FakeContextProvider()
    capability = Capability(name="memory", options={"namespace": "test"})

    profile = AgentProfile(
        name="knowledge_assistant",
        model=ModelConfigRef(name="default"),
        system_prompt=system_prompt,
        tools=(cast(Tool, tool),),
        permission_rules=(cast(PermissionRule, permission_rule),),
        context_providers=(cast(ContextProvider, context_provider),),
        skill_summaries=("test-skill: Test guidance.",),
        capabilities=(capability,),
    )

    assert profile.model.name == "default"
    assert profile.system_prompt() == "You are a test agent."
    assert profile.tools == (tool,)
    assert profile.permission_rules == (permission_rule,)
    assert profile.context_providers == (context_provider,)
    assert profile.skill_summaries == ("test-skill: Test guidance.",)
    assert profile.capabilities == (capability,)


@pytest.mark.parametrize("name", ["", "KnowledgeAssistant", "knowledge-assistant"])
def test_profile_rejects_invalid_names(name: str) -> None:
    """Profile 名称必须使用稳定的小写 snake_case 标识符。

    Profile names must use stable lowercase snake_case identifiers.
    """

    with pytest.raises(ValidationError):
        AgentProfile(
            name=name,
            model=ModelConfigRef(name="default"),
            system_prompt=system_prompt,
        )
