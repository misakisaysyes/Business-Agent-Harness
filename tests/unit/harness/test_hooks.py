"""Hook 事件、注册、阻断和错误策略测试。

Tests for hook events, registration, blocking, and failure policy.
"""

import logging
from typing import cast

import pytest
from tests.fakes import FakeModel, FakeSequenceModel

from harness.agent_loop import create_agent_loop
from harness.hooks import (
    HookDecision,
    HookEvent,
    HookEventType,
    HookExecutionError,
    HookFailureMode,
    HookRegistry,
    HookResult,
    LargeOutputWarningHook,
    PermissionCheckHook,
    PostToolUse,
    PreToolUse,
    Stop,
    StopMetricsHook,
    ToolCallLoggingHook,
    UserPromptSubmit,
)
from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider
from harness.permissions import PermissionDecision, PermissionResult
from harness.state import AgentState, AgentStopReason
from harness.tool_use import Tool, ToolInput
from services.checkpoint import create_in_memory_checkpointer


class EchoInput(ToolInput):
    """测试 Echo Tool 输入。

    Test input for the echo tool.
    """

    text: str


class RecordingEchoTool:
    """记录执行次数的测试 Tool。

    Test tool recording its invocation count.
    """

    name = "echo"
    description = "Echo text."
    input_schema = EchoInput
    concurrency_group = None

    def __init__(self) -> None:
        self.calls: list[ToolUse] = []

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        self.calls.append(tool_use)
        tool_input = EchoInput.model_validate(tool_use.input)
        return ToolResult(tool_use_id=tool_use.id, content=tool_input.text)


class RecordingHook:
    """记录 Hook 顺序并返回可配置结果。

    Record hook order and return a configurable result.
    """

    def __init__(
        self,
        name: str,
        events: frozenset[HookEventType],
        calls: list[str],
        result: HookResult | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.calls = calls
        self.result = result

    async def handle(self, event: HookEvent) -> HookResult | None:
        self.calls.append(f"{self.name}:{event.event_type.value}")
        return self.result


class FailingHook:
    """始终抛出异常的测试 Hook。

    Test hook that always raises an exception.
    """

    name = "failing"
    events = frozenset({HookEventType.USER_PROMPT_SUBMIT})

    async def handle(self, event: HookEvent) -> HookResult | None:
        raise RuntimeError("sensitive hook details")


def state() -> AgentState:
    """创建 Hook 测试状态。

    Create state used by hook tests.
    """

    return {
        "thread_id": "hook-thread",
        "messages": [Message(role=MessageRole.USER, content="hello")],
    }


def pre_tool_event(permission_granted: bool = True) -> PreToolUse:
    """创建 PreToolUse 测试事件。

    Create a PreToolUse test event.
    """

    decision = PermissionDecision.ALLOW if permission_granted else PermissionDecision.DENY
    return PreToolUse(
        state=state(),
        tool_use=ToolUse(id="tool-1", name="echo", input={"text": "secret"}),
        permission_result=PermissionResult(decision=decision, reason="test permission"),
        permission_granted=permission_granted,
    )


async def test_hooks_run_in_registration_order_and_stop_on_block() -> None:
    """Hook 应按注册顺序运行，并在首个 BLOCK 后停止。

    Hooks should run in registration order and stop after the first BLOCK.
    """

    calls: list[str] = []
    registry = HookRegistry(
        (
            RecordingHook("first", frozenset({HookEventType.PRE_TOOL_USE}), calls),
            RecordingHook(
                "blocker",
                frozenset({HookEventType.PRE_TOOL_USE}),
                calls,
                HookResult(decision=HookDecision.BLOCK, reason="blocked for test"),
            ),
            RecordingHook("never", frozenset({HookEventType.PRE_TOOL_USE}), calls),
        )
    )

    result = await registry.dispatch(pre_tool_event())

    assert result.blocked
    assert result.hook_name == "blocker"
    assert calls == ["first:PreToolUse", "blocker:PreToolUse"]


async def test_hook_failure_is_recorded_and_configured_to_continue_or_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hook 异常应记录，并按照 failure mode 继续或抛出。

    Hook failures should be recorded and either continue or raise by configuration.
    """

    event = UserPromptSubmit(state=state(), message=state()["messages"][0])
    calls: list[str] = []
    continuing = HookRegistry(
        (
            FailingHook(),
            RecordingHook("after", frozenset({HookEventType.USER_PROMPT_SUBMIT}), calls),
        ),
        failure_mode=HookFailureMode.CONTINUE,
    )

    with caplog.at_level(logging.ERROR, logger="harness.hooks"):
        result = await continuing.dispatch(event)

    assert calls == ["after:UserPromptSubmit"]
    assert result.errors[0].hook_name == "failing"
    assert "hook execution failed" in caplog.text
    assert "sensitive hook details" not in caplog.text

    raising = HookRegistry((FailingHook(),), failure_mode=HookFailureMode.RAISE)
    with pytest.raises(HookExecutionError, match="failing"):
        await raising.dispatch(event)


async def test_builtin_hooks_bridge_permission_warn_safely_and_count_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """内置 Hook 应桥接权限、安全记录日志、告警并统计 Stop。

    Built-in hooks should bridge permissions, log safely, warn, and count stops.
    """

    permission_hook = PermissionCheckHook()
    blocked = await permission_hook.handle(pre_tool_event(permission_granted=False))
    assert blocked is not None
    assert blocked.decision is HookDecision.BLOCK

    secret = "secret-content-must-not-appear"
    tool_use = ToolUse(id="tool-log", name="echo", input={"text": secret})
    tool_result = ToolResult(tool_use_id=tool_use.id, content=secret)
    permission = PermissionResult(decision=PermissionDecision.ALLOW, reason="allowed")
    logging_hook = ToolCallLoggingHook()
    warning_hook = LargeOutputWarningHook(max_chars=5)

    with caplog.at_level(logging.INFO, logger="harness.hooks"):
        await logging_hook.handle(
            PreToolUse(
                state=state(),
                tool_use=tool_use,
                permission_result=permission,
                permission_granted=True,
            )
        )
        await logging_hook.handle(
            PostToolUse(state=state(), tool_use=tool_use, tool_result=tool_result)
        )
        await warning_hook.handle(
            PostToolUse(state=state(), tool_use=tool_use, tool_result=tool_result)
        )

    assert "tool call started" in caplog.text
    assert "tool call finished" in caplog.text
    assert "tool output exceeds warning threshold" in caplog.text
    assert secret not in caplog.text

    metrics = StopMetricsHook()
    await metrics.handle(Stop(state=state(), reason=AgentStopReason.COMPLETED))
    assert metrics.counts[AgentStopReason.COMPLETED] == 1


def test_graph_pre_tool_hook_blocks_execution_and_returns_tool_result() -> None:
    """PreToolUse BLOCK 应阻止 Tool 执行并向模型回注错误结果。

    A PreToolUse BLOCK should skip execution and inject an error result into the model.
    """

    tool_use = ToolUse(id="echo-blocked", name="echo", input={"text": "hello"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="工具调用已阻断"),
        ]
    )
    tool = RecordingEchoTool()
    stop_calls: list[str] = []
    blocker = RecordingHook(
        "blocker",
        frozenset({HookEventType.PRE_TOOL_USE}),
        [],
        HookResult(decision=HookDecision.BLOCK, reason="business hook blocked tool"),
    )
    stop_hook = RecordingHook("stop", frozenset({HookEventType.STOP}), stop_calls)
    loop = create_agent_loop(
        cast(ModelProvider, model),
        lambda: "test",
        create_in_memory_checkpointer(),
        tools=(cast(Tool, tool),),
        hooks=(blocker, stop_hook),
    )

    result = loop.invoke(state())

    assert tool.calls == []
    blocked_result = model.sync_requests[1].messages[-1].tool_results[0]
    assert blocked_result.is_error
    assert "hook_blocked" in str(blocked_result.content)
    assert result["messages"][-1].content == "工具调用已阻断"
    assert stop_calls == ["stop:Stop"]


def test_post_tool_hook_observes_once_without_reexecuting_tool() -> None:
    """PostToolUse 应观察一次结果，并且不能导致 Tool 重复执行。

    PostToolUse should observe one result without causing repeated tool execution.
    """

    tool_use = ToolUse(id="echo-ok", name="echo", input={"text": "hello"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="完成"),
        ]
    )
    tool = RecordingEchoTool()
    calls: list[str] = []
    observer = RecordingHook(
        "observer",
        frozenset({HookEventType.POST_TOOL_USE, HookEventType.STOP}),
        calls,
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        lambda: "test",
        create_in_memory_checkpointer(),
        tools=(cast(Tool, tool),),
        hooks=(observer,),
    )

    loop.invoke(state())

    assert tool.calls == [tool_use]
    assert calls == ["observer:PostToolUse", "observer:Stop"]


def test_user_prompt_hook_can_stop_before_model_invocation() -> None:
    """UserPromptSubmit BLOCK 应在模型调用前结束 Agent Turn。

    A UserPromptSubmit BLOCK should end the agent turn before model invocation.
    """

    model = FakeModel(Message(role=MessageRole.ASSISTANT, content="should not run"))
    prompt_blocker = RecordingHook(
        "prompt_blocker",
        frozenset({HookEventType.USER_PROMPT_SUBMIT}),
        [],
        HookResult(decision=HookDecision.BLOCK, reason="prompt blocked"),
    )
    stop_calls: list[str] = []
    stop_hook = RecordingHook("stop", frozenset({HookEventType.STOP}), stop_calls)
    loop = create_agent_loop(
        cast(ModelProvider, model),
        lambda: "test",
        create_in_memory_checkpointer(),
        hooks=(prompt_blocker, stop_hook),
    )

    result = loop.invoke(state())

    assert model.sync_requests == []
    assert result["stop_reason"] is AgentStopReason.HOOK_BLOCKED
    assert stop_calls == ["stop:Stop"]
