"""最小 Graph 和 Agent Loop 测试。

Tests for the minimal graph and agent loop.
"""

from typing import cast

import pytest
from tests.fakes import FakeModel, FakeSequenceModel

from harness.agent_loop import create_agent_loop, get_permission_request
from harness.context import ContextFragment, ContextProvider
from harness.error_recovery import ErrorRecoveryPolicy, OutputTokenRecoveryError
from harness.graph import (
    COMPACT_NODE,
    FINAL_NODE,
    MODEL_NODE,
    PERMISSION_NODE,
    PREPARE_NODE,
    TOOL_NODE,
    build_agent_graph,
)
from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider
from harness.permissions import (
    PermissionApproval,
    PermissionDecision,
    PermissionResult,
    PermissionRule,
)
from harness.state import AgentState, AgentStopReason
from harness.tool_use import Tool, ToolInput
from services.checkpoint import create_in_memory_checkpointer


class RecordingCalculatorInput(ToolInput):
    """测试 Calculator 的参数 Schema。

    Input schema for the test calculator.
    """

    expression: str


class RecordingCalculator:
    """记录调用并返回固定计算结果的测试 Tool。

    Test tool that records calls and returns a fixed calculation result.
    """

    name = "calculator"
    description = "Evaluate a safe arithmetic expression."
    input_schema = RecordingCalculatorInput
    concurrency_group = None

    def __init__(self) -> None:
        self.calls: list[ToolUse] = []

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """记录 ToolUse 并返回结果 5。

        Record the tool use and return the result 5.
        """

        self.calls.append(tool_use)
        return ToolResult(tool_use_id=tool_use.id, content=5)


class AskCalculatorPermissionRule:
    """每次 Calculator 调用都请求用户确认。

    Ask the user to approve every calculator call.
    """

    name = "ask_calculator"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != "calculator":
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason="calculator requires test approval",
        )


class CurrentTaskContextProvider:
    """把最近用户任务注入 Prompt 的测试 Context Provider。"""

    name = "current_task"

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                key="current_task",
                title="Current Task",
                content=state["messages"][-1].content,
                priority=100,
            ),
        )


def system_prompt() -> str:
    """返回固定测试 System Prompt。

    Return a fixed test system prompt.
    """

    return "You are a test assistant."


def initial_state() -> AgentState:
    """创建包含一条 User Message 的初始状态。

    Create initial state containing one user message.
    """

    return {
        "thread_id": "thread-001",
        "messages": [Message(role=MessageRole.USER, content="你好")],
    }


def test_agent_graph_contains_complete_loop_nodes() -> None:
    """M2-1 Graph 应包含完整 Agent Loop 节点。

    The M2-1 graph should contain all complete agent-loop nodes.
    """

    fake_model = FakeModel(Message(role=MessageRole.ASSISTANT, content="你好"))
    graph = build_agent_graph(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )

    assert set(graph.nodes) == {
        "__start__",
        PREPARE_NODE,
        COMPACT_NODE,
        MODEL_NODE,
        PERMISSION_NODE,
        TOOL_NODE,
        FINAL_NODE,
    }


def test_invoke_appends_model_response_without_mutating_input() -> None:
    """同步调用应追加响应且不修改输入 State。

    Synchronous invocation should append the response without mutating the input state.
    """

    response = Message(role=MessageRole.ASSISTANT, content="固定回答")
    fake_model = FakeModel(response)
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )
    state = initial_state()
    original_messages = list(state["messages"])

    result = loop.invoke(state)

    assert result["messages"] == [*original_messages, response]
    assert state["messages"] == original_messages
    assembled_prompt = fake_model.sync_requests[0].system_prompt
    assert "# Core Instructions" in assembled_prompt
    assert "# Business Instructions\nYou are a test assistant." in assembled_prompt
    assert fake_model.sync_requests[0].messages == tuple(original_messages)
    assert fake_model.async_requests == []
    assert result["iteration_count"] == 1
    assert result["stop_reason"] is AgentStopReason.COMPLETED


def test_agent_loop_injects_selected_context_into_model_request() -> None:
    """Graph 应把 ContextManager 的选择结果写入模型 System Prompt。"""

    fake_model = FakeModel(Message(role=MessageRole.ASSISTANT, content="done"))
    provider = cast(ContextProvider, CurrentTaskContextProvider())
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
        context_providers=(provider,),
    )

    loop.invoke(initial_state())

    prompt = fake_model.sync_requests[0].system_prompt
    assert "# Runtime Context" in prompt
    assert "## Current Task\n你好" in prompt


def test_tool_result_is_injected_before_second_model_call() -> None:
    """ToolResult 应回注消息后再进行第二次模型调用。

    The tool result should be injected before the second model invocation.
    """

    tool_use = ToolUse(id="tool-001", name="calculator", input={"expression": "2 + 3"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="结果是 5"),
        ]
    )
    calculator = RecordingCalculator()
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, calculator),),
    )

    result = loop.invoke(initial_state())

    assert calculator.calls == [tool_use]
    assert len(model.sync_requests) == 2
    tool_message = model.sync_requests[1].messages[-1]
    assert tool_message.role is MessageRole.TOOL
    assert tool_message.tool_results[0].tool_use_id == tool_use.id
    assert tool_message.tool_results[0].content == 5
    assert result["messages"][-1].content == "结果是 5"
    assert result["iteration_count"] == 2
    assert result["stop_reason"] is AgentStopReason.COMPLETED


def test_required_tool_applies_only_to_first_model_call() -> None:
    """强制工具只应用于当前 Turn 第一次模型调用，避免重复调用。"""

    tool_use = ToolUse(id="tool-required", name="calculator", input={"expression": "2 + 3"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="结果是 5"),
        ]
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, RecordingCalculator()),),
    )
    state = initial_state()
    state["metadata"] = {"required_tool": "calculator"}

    loop.invoke(state)

    assert model.sync_requests[0].required_tool == "calculator"
    assert model.sync_requests[1].required_tool is None


def test_max_iterations_stops_before_repeating_tool_execution() -> None:
    """达到最大模型调用次数后不得继续执行新的 ToolUse。

    A new tool use must not execute after the maximum model-call count is reached.
    """

    first_tool_use = ToolUse(id="tool-001", name="calculator", input={"expression": "2 + 3"})
    second_tool_use = ToolUse(id="tool-002", name="calculator", input={"expression": "5 + 1"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(first_tool_use,)),
            Message(role=MessageRole.ASSISTANT, tool_uses=(second_tool_use,)),
        ]
    )
    calculator = RecordingCalculator()
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, calculator),),
        max_iterations=2,
    )

    result = loop.invoke(initial_state())

    assert calculator.calls == [first_tool_use]
    assert len(model.sync_requests) == 2
    assert result["iteration_count"] == 2
    assert result["stop_reason"] is AgentStopReason.MAX_ITERATIONS


def test_permission_interrupt_resumes_and_executes_approved_tool_once() -> None:
    """ASK 应暂停 Graph，批准后只执行一次 Tool 并继续模型循环。

    ASK should pause the graph and execute the approved tool exactly once after resume.
    """

    tool_use = ToolUse(id="tool-ask", name="calculator", input={"expression": "2 + 3"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="批准后结果是 5"),
        ]
    )
    calculator = RecordingCalculator()
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, calculator),),
        permission_rules=(cast(PermissionRule, AskCalculatorPermissionRule()),),
    )

    paused = loop.invoke(initial_state())
    request = get_permission_request(paused)

    assert request is not None
    assert request.requests[0].tool_use_id == tool_use.id
    assert calculator.calls == []
    assert len(model.sync_requests) == 1

    result = loop.resume("thread-001", True)

    assert calculator.calls == [tool_use]
    assert len(model.sync_requests) == 2
    capability_state = result.get("capability_state")
    assert capability_state is not None
    permission_history = capability_state["permission_history"]
    assert isinstance(permission_history, list)
    assert permission_history[-1]["tool_use_id"] == tool_use.id
    assert permission_history[-1]["allowed"] is True
    assert result["messages"][-1].content == "批准后结果是 5"
    assert result["stop_reason"] is AgentStopReason.COMPLETED


def test_permission_interrupt_denial_returns_tool_error_without_execution() -> None:
    """拒绝审批后不得执行 Tool，并应把错误 ToolResult 回注模型。

    Denying approval should skip execution and inject an error tool result into the model.
    """

    tool_use = ToolUse(id="tool-denied", name="calculator", input={"expression": "2 + 3"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="用户拒绝了操作"),
        ]
    )
    calculator = RecordingCalculator()
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, calculator),),
        permission_rules=(cast(PermissionRule, AskCalculatorPermissionRule()),),
    )

    paused = loop.invoke(initial_state())
    assert get_permission_request(paused) is not None

    result = loop.resume("thread-001", False)

    assert calculator.calls == []
    tool_message = model.sync_requests[1].messages[-1]
    denied_result = tool_message.tool_results[0]
    assert denied_result.tool_use_id == tool_use.id
    assert denied_result.is_error
    assert "permission_denied" in str(denied_result.content)
    assert result["messages"][-1].content == "用户拒绝了操作"


def test_permission_batch_can_approve_and_deny_individual_tool_uses() -> None:
    """一批 ASK ToolUse 应支持逐项批准，并保持 ToolResult 顺序。

    A batch of ASK tool uses should support per-call approval and preserve result order.
    """

    first = ToolUse(id="tool-first", name="calculator", input={"expression": "2 + 3"})
    second = ToolUse(id="tool-second", name="calculator", input={"expression": "5 + 1"})
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(first, second)),
            Message(role=MessageRole.ASSISTANT, content="已处理批量审批"),
        ]
    )
    calculator = RecordingCalculator()
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(cast(Tool, calculator),),
        permission_rules=(cast(PermissionRule, AskCalculatorPermissionRule()),),
    )

    paused = loop.invoke(initial_state())
    request = get_permission_request(paused)
    assert request is not None
    assert tuple(item.tool_use_id for item in request.requests) == (first.id, second.id)

    loop.resume(
        "thread-001",
        PermissionApproval(decisions={first.id: True, second.id: False}),
    )

    assert calculator.calls == [first]
    results = model.sync_requests[1].messages[-1].tool_results
    assert tuple(result.tool_use_id for result in results) == (first.id, second.id)
    assert not results[0].is_error
    assert results[1].is_error


def test_cancel_signal_stops_before_model_invocation() -> None:
    """预先设置取消信号时不得调用模型。

    A pre-set cancellation signal should stop before invoking the model.
    """

    fake_model = FakeModel(Message(role=MessageRole.ASSISTANT, content="不应返回"))
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )
    state = initial_state()
    state["cancel_requested"] = True

    result = loop.invoke(state)

    assert fake_model.sync_requests == []
    assert result["messages"] == state["messages"]
    assert result["iteration_count"] == 0
    assert result["stop_reason"] is AgentStopReason.CANCELLED


def test_truncated_model_output_retries_once_with_a_higher_limit() -> None:
    """输出被截断时应提高上限重做当前模型请求。"""

    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content="partial",
                provider_metadata={"finish_reason": "length"},
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="complete answer",
                provider_metadata={"finish_reason": "stop"},
            ),
        ]
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        error_recovery=ErrorRecoveryPolicy(
            initial_max_output_tokens=100,
            max_output_tokens=200,
            output_token_multiplier=2,
            max_output_retries=1,
        ),
    )

    result = loop.invoke(initial_state())

    assert [item.max_output_tokens for item in model.sync_requests] == [100, 200]
    assert result["messages"][-1].content == "complete answer"


def test_repeated_output_truncation_stops_after_the_configured_retry() -> None:
    """提高输出上限后仍截断时应明确失败，不能无限重试。"""

    truncated = Message(
        role=MessageRole.ASSISTANT,
        content="partial",
        provider_metadata={"finish_reason": "length"},
    )
    model = FakeSequenceModel([truncated, truncated])
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        error_recovery=ErrorRecoveryPolicy(
            initial_max_output_tokens=100,
            max_output_tokens=200,
            output_token_multiplier=2,
            max_output_retries=1,
        ),
    )

    with pytest.raises(OutputTokenRecoveryError, match="remained truncated"):
        loop.invoke(initial_state())

    assert len(model.sync_requests) == 2


@pytest.mark.asyncio
async def test_ainvoke_uses_async_model_interface() -> None:
    """异步调用应使用 ModelProvider 的异步接口。

    Asynchronous invocation should use the model provider's asynchronous interface.
    """

    response = Message(role=MessageRole.ASSISTANT, content="异步回答")
    fake_model = FakeModel(response)
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )

    result = await loop.ainvoke(initial_state())

    assert result["messages"][-1] == response
    assert len(fake_model.async_requests) == 1
    assert fake_model.sync_requests == []


def test_stream_emits_final_state_and_invokes_model_once() -> None:
    """流式执行应输出最终 State 且只调用一次模型。

    Streaming should emit the final state and invoke the model exactly once.
    """

    response = Message(role=MessageRole.ASSISTANT, content="流式回答")
    fake_model = FakeModel(response)
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )

    snapshots = list(loop.stream(initial_state()))

    assert snapshots[-1]["messages"][-1] == response
    assert len(fake_model.sync_requests) == 1


def test_non_assistant_model_response_is_rejected() -> None:
    """Model Provider 返回非 Assistant Message 时应明确失败。

    A model provider returning a non-assistant message should fail explicitly.
    """

    fake_model = FakeModel(Message(role=MessageRole.USER, content="错误角色"))
    loop = create_agent_loop(
        cast(ModelProvider, fake_model),
        system_prompt,
        create_in_memory_checkpointer(),
    )

    with pytest.raises(ValueError, match="assistant message"):
        loop.invoke(initial_state())
