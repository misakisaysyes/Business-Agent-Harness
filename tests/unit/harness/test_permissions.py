"""Permission Pipeline 单元测试。

Unit tests for the permission pipeline.
"""

from harness.messages import Message, MessageRole, ToolUse
from harness.permissions import (
    PermissionApproval,
    PermissionDecision,
    PermissionPipeline,
    PermissionResult,
)
from harness.state import AgentState


class StaticPermissionRule:
    """返回固定决策并记录执行顺序的测试规则。

    Test rule returning a fixed decision and recording evaluation order.
    """

    def __init__(
        self,
        name: str,
        decision: PermissionDecision,
        calls: list[str],
    ) -> None:
        self.name = name
        self.decision = decision
        self.calls = calls

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionDecision:
        self.calls.append(self.name)
        return self.decision


def state() -> AgentState:
    """创建 Permission 测试状态。

    Create state used by permission tests.
    """

    return {
        "thread_id": "permission-thread",
        "messages": [Message(role=MessageRole.USER, content="test")],
    }


async def test_passthrough_continues_to_first_conclusive_rule() -> None:
    """PASSTHROUGH 应继续执行，首个明确结果应终止规则链。

    PASSTHROUGH should continue and the first conclusive result should stop the chain.
    """

    calls: list[str] = []
    pipeline = PermissionPipeline(
        (
            StaticPermissionRule("pass", PermissionDecision.PASSTHROUGH, calls),
            StaticPermissionRule("allow", PermissionDecision.ALLOW, calls),
            StaticPermissionRule("deny", PermissionDecision.DENY, calls),
        ),
        known_tool_names=("safe_tool",),
    )

    result = await pipeline.evaluate(ToolUse(id="1", name="safe_tool"), state())

    assert result.decision is PermissionDecision.ALLOW
    assert result.rule_name == "allow"
    assert calls == ["pass", "allow"]


async def test_pipeline_preserves_deny_and_ask_decisions() -> None:
    """Permission Pipeline 应保留 DENY 和 ASK 明确决策。

    The permission pipeline should preserve explicit DENY and ASK decisions.
    """

    deny = PermissionPipeline(
        (StaticPermissionRule("deny", PermissionDecision.DENY, []),),
        known_tool_names=("tool",),
    )
    ask = PermissionPipeline(
        (StaticPermissionRule("ask", PermissionDecision.ASK, []),),
        known_tool_names=("tool",),
    )
    tool_use = ToolUse(id="1", name="tool")

    assert (await deny.evaluate(tool_use, state())).decision is PermissionDecision.DENY
    assert (await ask.evaluate(tool_use, state())).decision is PermissionDecision.ASK


async def test_unknown_and_unmatched_tools_are_denied_by_default() -> None:
    """未知 Tool 和未被已配置规则处理的 Tool 都应默认拒绝。

    Unknown tools and tools unmatched by configured rules should be denied by default.
    """

    calls: list[str] = []
    pass_rule = StaticPermissionRule("pass", PermissionDecision.PASSTHROUGH, calls)
    pipeline = PermissionPipeline((pass_rule,), known_tool_names=("known",))

    unknown = await pipeline.evaluate(ToolUse(id="1", name="unknown"), state())
    unmatched = await pipeline.evaluate(ToolUse(id="2", name="known"), state())

    assert unknown.decision is PermissionDecision.DENY
    assert unknown.rule_name == "default_deny_unknown_tool"
    assert unmatched.decision is PermissionDecision.DENY
    assert unmatched.rule_name == "default_deny_unmatched_tool"


def test_approval_supports_global_and_per_tool_decisions() -> None:
    """审批响应应支持全部批准和逐 ToolUse 决策。

    Approval responses should support global and per-tool-use decisions.
    """

    global_approval = PermissionApproval(approved=True)
    individual = PermissionApproval(decisions={"read": True, "write": False})

    assert global_approval.is_approved("any")
    assert individual.is_approved("read")
    assert not individual.is_approved("write")


def test_permission_result_carries_reason_and_rule_name() -> None:
    """结构化结果应保留决策原因和规则来源。

    A structured result should retain its reason and source rule.
    """

    result = PermissionResult(
        decision=PermissionDecision.ALLOW,
        reason="safe",
        rule_name="test",
    )

    assert result.reason == "safe"
    assert result.rule_name == "test"
