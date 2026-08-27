"""Tool Registry、dispatch 和执行边界测试。

Tests for the tool registry, dispatch, and execution boundaries.
"""

import asyncio

import pytest

from harness.messages import ToolResult, ToolUse
from harness.tool_use import DuplicateToolError, ToolErrorCode, ToolInput, ToolRegistry


class EchoInput(ToolInput):
    """EchoTool 的参数。

    Input for EchoTool.
    """

    text: str


class EchoTool:
    """返回输入文本的测试 Tool。

    Test tool returning its input text.
    """

    name = "echo"
    description = "Echo text."
    input_schema = EchoInput

    def __init__(
        self,
        concurrency_group: str | None = None,
        delay: float = 0,
    ) -> None:
        self.concurrency_group = concurrency_group
        self.delay = delay
        self.active_calls = 0
        self.max_active_calls = 0

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delay)
            tool_input = EchoInput.model_validate(tool_use.input)
            return ToolResult(tool_use_id=tool_use.id, content=tool_input.text)
        finally:
            self.active_calls -= 1


class RecordingTool(EchoTool):
    """记录多个 Tool 共享资源时的执行顺序。

    Record execution order when multiple tools share one resource.
    """

    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(concurrency_group="shared-resource", delay=0.01)
        self.name = name
        self.events = events

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        self.events.append(f"{self.name}:start")
        result = await super().ainvoke(tool_use)
        self.events.append(f"{self.name}:end")
        return result


class FailingTool(EchoTool):
    """始终失败的测试 Tool。

    Test tool that always fails.
    """

    name = "failing"

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        raise RuntimeError("boom")


class MismatchedTool(EchoTool):
    """返回错误 ToolUse ID 的测试 Tool。

    Test tool returning the wrong tool-use ID.
    """

    name = "mismatched"

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        return ToolResult(tool_use_id="wrong-id", content="wrong")


def tool_use(identifier: str, name: str = "echo", text: str = "hello") -> ToolUse:
    """创建标准测试 ToolUse。

    Create a standard test tool use.
    """

    return ToolUse(id=identifier, name=name, input={"text": text})


def test_registry_exposes_schema_and_rejects_duplicate_names() -> None:
    """Registry 应暴露参数 Schema 并拒绝重复 Tool 名称。

    The registry should expose input schemas and reject duplicate tool names.
    """

    registry = ToolRegistry((EchoTool(),))

    assert registry.names() == ("echo",)
    assert registry.definitions()[0].name == "echo"
    assert "text" in registry.definitions()[0].parameters["properties"]

    with pytest.raises(DuplicateToolError, match="echo"):
        registry.register(EchoTool())


@pytest.mark.asyncio
async def test_dispatch_maps_unknown_invalid_and_failed_calls_to_paired_errors() -> None:
    """未知、参数错误和异常都应变成配对 ToolResult。

    Unknown, invalid, and failed calls should become paired tool results.
    """

    registry = ToolRegistry((EchoTool(), FailingTool(), MismatchedTool()))
    calls = (
        tool_use("unknown-1", name="missing"),
        ToolUse(id="invalid-1", name="echo", input={}),
        tool_use("failed-1", name="failing"),
        tool_use("mismatch-1", name="mismatched"),
    )

    results = await registry.adispatch_many(calls)

    assert tuple(result.tool_use_id for result in results) == tuple(call.id for call in calls)
    assert all(result.is_error for result in results)
    assert results[0].content["error"] == ToolErrorCode.UNKNOWN_TOOL
    assert results[1].content["error"] == ToolErrorCode.INVALID_INPUT
    assert results[2].content["error"] == ToolErrorCode.EXECUTION_FAILED
    assert "RuntimeError: boom" in str(results[2].content["message"])
    assert results[3].content["error"] == ToolErrorCode.RESULT_ID_MISMATCH
    assert all(result.content["retryable"] is False for result in results)


@pytest.mark.asyncio
async def test_dispatch_maps_timeout_and_truncates_large_output() -> None:
    """超时应返回错误，过大输出应被截断。

    Timeouts should return errors and oversized output should be truncated.
    """

    slow_registry = ToolRegistry((EchoTool(delay=0.05),), timeout_seconds=0.001)
    timeout_result = await slow_registry.dispatch(tool_use("slow-1"))

    assert timeout_result.is_error
    assert timeout_result.content["error"] == ToolErrorCode.TIMEOUT
    assert "timed out" in str(timeout_result.content["message"])

    short_registry = ToolRegistry((EchoTool(),), max_output_chars=32)
    truncated = await short_registry.dispatch(tool_use("long-1", text="x" * 100))

    assert not truncated.is_error
    assert isinstance(truncated.content, str)
    assert len(truncated.content) == 32
    assert truncated.content.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_parallel_safe_calls_overlap_but_same_group_is_serialized() -> None:
    """并行安全调用应重叠，同组调用必须串行。

    Parallel-safe calls should overlap while calls in the same group are serialized.
    """

    parallel_tool = EchoTool(delay=0.01)
    parallel_registry = ToolRegistry((parallel_tool,))
    await parallel_registry.adispatch_many((tool_use("p1"), tool_use("p2")))

    serial_tool = EchoTool(concurrency_group="writes", delay=0.01)
    serial_registry = ToolRegistry((serial_tool,))
    await serial_registry.adispatch_many((tool_use("s1"), tool_use("s2")))

    assert parallel_tool.max_active_calls == 2
    assert serial_tool.max_active_calls == 1


@pytest.mark.asyncio
async def test_different_tools_sharing_a_resource_group_are_serialized() -> None:
    """不同 Tool 访问同类共享资源时也必须串行。

    Different tools accessing the same shared resource must also be serialized.
    """

    events: list[str] = []
    first_tool = RecordingTool("first", events)
    second_tool = RecordingTool("second", events)
    registry = ToolRegistry((first_tool, second_tool))

    await registry.adispatch_many(
        (
            tool_use("first-call", name="first"),
            tool_use("second-call", name="second"),
        )
    )

    assert events == ["first:start", "first:end", "second:start", "second:end"]


def test_synchronous_dispatch_preserves_input_order() -> None:
    """同步 dispatch 应保持 ToolUse 的输入顺序。

    Synchronous dispatch should preserve tool-use input order.
    """

    registry = ToolRegistry((EchoTool(),))
    calls = (tool_use("first", text="1"), tool_use("second", text="2"))

    results = registry.dispatch_many(calls)

    assert tuple(result.tool_use_id for result in results) == ("first", "second")
    assert tuple(result.content for result in results) == ("1", "2")
