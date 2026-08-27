"""Permission 决策、规则、审批和执行管线。

Permission decisions, rules, approvals, and execution pipeline.
"""

import asyncio
from collections.abc import Coroutine, Iterable, Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from harness.messages import ToolUse
from harness.state import AgentState


class PermissionDecision(StrEnum):
    """Permission Rule 可以返回的标准决策。

    Standard decisions returned by a permission rule.
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    PASSTHROUGH = "passthrough"


class PermissionResult(BaseModel):
    """一条 Permission Rule 的结构化决策结果。

    Structured decision returned by a permission rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PermissionDecision
    reason: str = Field(min_length=1)
    rule_name: str | None = None


class PermissionRequestItem(BaseModel):
    """一次需要用户审批的 ToolUse。

    One tool use requiring user approval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_use_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str = Field(min_length=1)


class PermissionRequest(BaseModel):
    """通过 LangGraph interrupt 暴露给调用端的批量审批请求。

    Batch approval request exposed to the caller through a LangGraph interrupt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_permission"] = "tool_permission"
    requests: tuple[PermissionRequestItem, ...] = Field(min_length=1)


class PermissionApproval(BaseModel):
    """调用端恢复 Graph 时提交的审批结果。

    Approval response supplied by the caller when resuming the graph.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool | None = None
    decisions: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_approval(self) -> "PermissionApproval":
        """要求提供全局决定或至少一个逐项决定。

        Require either one global decision or at least one per-call decision.
        """

        if self.approved is None and not self.decisions:
            raise ValueError("approval must include approved or decisions")
        return self

    @classmethod
    def from_resume(cls, value: object) -> "PermissionApproval":
        """解析 LangGraph Command.resume 返回的值。

        Parse the value returned from LangGraph Command.resume.
        """

        if isinstance(value, bool):
            return cls(approved=value)
        return cls.model_validate(value)

    def is_approved(self, tool_use_id: str) -> bool:
        """返回一个 ToolUse 的审批结果。

        Return the approval decision for one tool use.
        """

        if tool_use_id in self.decisions:
            return self.decisions[tool_use_id]
        if self.approved is not None:
            return self.approved
        raise ValueError(f"missing approval for tool use: {tool_use_id}")


PermissionRuleResult = PermissionDecision | PermissionResult | None


@runtime_checkable
class PermissionRule(Protocol):
    """在 Tool 执行前评估权限的通用协议。

    Shared protocol for evaluating permission before tool execution.
    """

    @property
    def name(self) -> str:
        """返回 Permission Rule 的唯一名称。

        Return the unique permission-rule name.
        """
        ...

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionRuleResult:
        """返回当前规则的决策；不适用时返回 PASSTHROUGH 或空值。

        Return this rule's decision, or PASSTHROUGH/no value when not applicable.
        """
        ...


def _normalize_rule_result(
    result: PermissionRuleResult,
    rule_name: str,
) -> PermissionResult:
    """把简写决策转换成带原因和规则名的完整结果。

    Convert a shorthand decision into a result with a reason and rule name.
    """

    if isinstance(result, PermissionResult):
        if result.rule_name is not None:
            return result
        return result.model_copy(update={"rule_name": rule_name})

    decision = result or PermissionDecision.PASSTHROUGH
    return PermissionResult(
        decision=decision,
        reason=f"permission rule returned {decision.value}",
        rule_name=rule_name,
    )


class PermissionPipeline:
    """按注册顺序执行 Permission Rule，并采用首个明确决策。

    Evaluate permission rules in registration order and use the first conclusive result.
    """

    def __init__(
        self,
        rules: Sequence[PermissionRule] = (),
        known_tool_names: Iterable[str] = (),
    ) -> None:
        self.rules = tuple(rules)
        self.known_tool_names = frozenset(known_tool_names)

    async def evaluate(self, tool_use: ToolUse, state: AgentState) -> PermissionResult:
        """评估一个 ToolUse，并安全处理未命中的默认情况。

        Evaluate one tool use and safely handle the unmatched default case.
        """

        for rule in self.rules:
            try:
                raw_result = await rule.evaluate(tool_use, state)
            except Exception as error:
                return PermissionResult(
                    decision=PermissionDecision.DENY,
                    reason=(
                        f"permission rule failed: {rule.name}: {type(error).__name__}: {error}"
                    ),
                    rule_name=rule.name,
                )

            result = _normalize_rule_result(raw_result, rule.name)
            if result.decision is PermissionDecision.PASSTHROUGH:
                continue
            return result

        if tool_use.name not in self.known_tool_names:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"unknown high-risk tool is denied: {tool_use.name}",
                rule_name="default_deny_unknown_tool",
            )

        if self.rules:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"no permission rule allowed tool: {tool_use.name}",
                rule_name="default_deny_unmatched_tool",
            )

        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason=(
                "registered tool is allowed when no permission rules are configured: "
                f"{tool_use.name}"
            ),
            rule_name="default_allow_registered_tool",
        )

    async def evaluate_many(
        self,
        tool_uses: Sequence[ToolUse],
        state: AgentState,
    ) -> tuple[PermissionResult, ...]:
        """按 ToolUse 输入顺序评估一批调用。

        Evaluate a batch of tool uses in input order.
        """

        results: list[PermissionResult] = []
        for tool_use in tool_uses:
            results.append(await self.evaluate(tool_use, state))
        return tuple(results)

    def evaluate_many_sync(
        self,
        tool_uses: Sequence[ToolUse],
        state: AgentState,
    ) -> tuple[PermissionResult, ...]:
        """从同步 Graph Node 执行异步 Permission Rule。

        Run asynchronous permission rules from a synchronous graph node.
        """

        coroutine: Coroutine[Any, Any, tuple[PermissionResult, ...]] = self.evaluate_many(
            tool_uses,
            state,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        coroutine.close()
        raise RuntimeError("use the asynchronous agent loop inside an active event loop")


__all__ = [
    "PermissionApproval",
    "PermissionDecision",
    "PermissionPipeline",
    "PermissionRequest",
    "PermissionRequestItem",
    "PermissionResult",
    "PermissionRule",
    "PermissionRuleResult",
]
