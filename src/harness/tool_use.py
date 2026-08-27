"""Tool 契约、注册、分发和安全执行。

Tool contracts, registration, dispatch, and safe execution.
"""

import asyncio
import json
from collections import defaultdict
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from jsonschema import SchemaError as JsonSchemaError
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from harness.messages import ToolResult, ToolUse

DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 12_000
TRUNCATION_SUFFIX = "... [truncated]"
ToolInputSchema = type[BaseModel] | dict[str, JsonValue]


class ToolInput(BaseModel):
    """所有 Tool 参数 Schema 的严格基类。

    Strict base class for all tool input schemas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDefinition(BaseModel):
    """发送给模型的厂商无关 Tool 定义。

    Provider-independent tool definition sent to a model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue]


class ToolStatePatch(BaseModel):
    """Tool 执行成功后写回 Agent State 的命名空间更新。

    Namespaced Agent State update emitted after a successful tool execution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    value: JsonValue


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """运行时提供、不会暴露给模型的 Tool 调用上下文。

    Runtime-provided tool context that is never exposed to the model.
    """

    thread_id: str
    metadata: dict[str, JsonValue]


@runtime_checkable
class Tool(Protocol):
    """业务 Tool 必须实现的异步执行协议。

    Asynchronous execution protocol implemented by business tools.
    """

    @property
    def name(self) -> str:
        """返回 Tool 的唯一名称。

        Return the unique tool name.
        """
        ...

    @property
    def description(self) -> str:
        """返回供模型理解的 Tool 描述。

        Return a model-facing description of the tool.
        """
        ...

    @property
    def input_schema(self) -> ToolInputSchema:
        """返回用于校验 ToolUse.input 的 Pydantic Schema。

        Return the Pydantic schema used to validate ToolUse.input.
        """
        ...

    @property
    def concurrency_group(self) -> str | None:
        """返回必须串行执行的分组；空值表示可并行。

        Return the serialization group, or no value when parallel-safe.
        """
        ...

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """异步执行一次 ToolUse 并返回配对结果。

        Execute one ToolUse asynchronously and return its paired result.
        """
        ...


@runtime_checkable
class ContextAwareTool(Protocol):
    """需要可信运行时上下文的可选 Tool 扩展协议。"""

    async def ainvoke_with_context(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """使用 Graph 提供的上下文执行 ToolUse。"""
        ...


@runtime_checkable
class StatefulTool(Protocol):
    """执行成功后可以产生 Capability State 更新的 Tool。

    Tool that can emit a capability-state update after successful execution.
    """

    def state_update(
        self,
        tool_use: ToolUse,
        tool_result: ToolResult,
    ) -> ToolStatePatch:
        """根据已执行 ToolUse 返回状态更新。

        Return a state update derived from an executed tool use.
        """
        ...


class DuplicateToolError(ValueError):
    """Tool Registry 中出现了重复名称。

    Raised when duplicate names are registered in a tool registry.
    """


class ToolErrorCode(StrEnum):
    """模型和客户端都可以稳定识别的 Tool 错误码。"""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_INPUT = "invalid_tool_input"
    TIMEOUT = "tool_timeout"
    EXECUTION_FAILED = "tool_execution_failed"
    RESULT_ID_MISMATCH = "tool_result_id_mismatch"


def _error_result(
    tool_use: ToolUse,
    code: ToolErrorCode,
    message: str,
) -> ToolResult:
    """创建始终与原 ToolUse 配对的错误结果。

    Create an error result always paired with the original tool use.
    """

    return ToolResult(
        tool_use_id=tool_use.id,
        content={
            "error": code.value,
            "message": message,
            # Tool 可能包含外部副作用，本期不做自动重试。
            # Tools may have side effects, so this phase never retries blindly.
            "retryable": False,
        },
        is_error=True,
    )


def _truncate_result(result: ToolResult, max_chars: int) -> ToolResult:
    """截断过大的 Tool 输出，同时保留错误标记和配对 ID。

    Truncate oversized tool output while preserving its error flag and paired ID.
    """

    serialized = (
        result.content
        if isinstance(result.content, str)
        else json.dumps(result.content, ensure_ascii=False, separators=(",", ":"))
    )
    if len(serialized) <= max_chars:
        return result

    kept_chars = max(0, max_chars - len(TRUNCATION_SUFFIX))
    return result.model_copy(update={"content": serialized[:kept_chars] + TRUNCATION_SUFFIX})


def _run_coroutine(
    coroutine: Coroutine[Any, Any, tuple[ToolResult, ...]],
) -> tuple[ToolResult, ...]:
    """从同步入口运行 Tool dispatch coroutine。

    Run the tool-dispatch coroutine from a synchronous entry point.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    coroutine.close()
    raise RuntimeError("use ToolRegistry.adispatch_many() inside an active event loop")


class ToolRegistry:
    """注册 Tool，并统一处理校验、调度和错误边界。

    Register tools and centralize validation, dispatch, and error boundaries.
    """

    def __init__(
        self,
        tools: Sequence[Tool] = (),
        timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if max_output_chars < len(TRUNCATION_SUFFIX):
            raise ValueError("max_output_chars is too small")

        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self._tools: dict[str, Tool] = {}

        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """注册一个 Tool，并拒绝重复名称。

        Register one tool and reject duplicate names.
        """

        if tool.name in self._tools:
            raise DuplicateToolError(f"tool is already registered: {tool.name}")
        if isinstance(tool.input_schema, dict):
            validator_for(tool.input_schema).check_schema(tool.input_schema)
        self._tools[tool.name] = tool

    def names(self) -> tuple[str, ...]:
        """返回排序后的 Tool 名称。

        Return registered tool names in sorted order.
        """

        return tuple(sorted(self._tools))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """返回可发送给模型的 Tool Schema。

        Return model-facing tool schemas.
        """

        return tuple(
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=(
                    tool.input_schema
                    if isinstance(tool.input_schema, dict)
                    else cast(dict[str, JsonValue], tool.input_schema.model_json_schema())
                ),
            )
            for tool in self._tools.values()
        )

    async def dispatch(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """校验并安全执行一次 ToolUse。

        Validate and safely execute one tool use.
        """

        tool = self._tools.get(tool_use.name)
        if tool is None:
            return _error_result(
                tool_use,
                ToolErrorCode.UNKNOWN_TOOL,
                f"unknown tool: {tool_use.name}",
            )

        try:
            if isinstance(tool.input_schema, dict):
                validator_for(tool.input_schema)(tool.input_schema).validate(tool_use.input)
                validated_input = tool_use.input
            else:
                validated = tool.input_schema.model_validate(tool_use.input)
                validated_input = cast(
                    dict[str, JsonValue],
                    validated.model_dump(mode="json"),
                )
        except (ValidationError, JsonSchemaValidationError, JsonSchemaError) as error:
            return _error_result(
                tool_use,
                ToolErrorCode.INVALID_INPUT,
                f"invalid tool input: {error}",
            )

        validated_use = tool_use.model_copy(
            update={"input": validated_input}
        )

        try:
            invocation = (
                tool.ainvoke_with_context(validated_use, context)
                if context is not None and isinstance(tool, ContextAwareTool)
                else tool.ainvoke(validated_use)
            )
            result = await asyncio.wait_for(invocation, timeout=self.timeout_seconds)
        except TimeoutError:
            return _error_result(
                tool_use,
                ToolErrorCode.TIMEOUT,
                f"tool timed out after {self.timeout_seconds:g}s",
            )
        except Exception as error:
            return _error_result(
                tool_use,
                ToolErrorCode.EXECUTION_FAILED,
                f"tool execution failed: {type(error).__name__}: {error}",
            )

        if result.tool_use_id != tool_use.id:
            return _error_result(
                tool_use,
                ToolErrorCode.RESULT_ID_MISMATCH,
                f"tool result ID mismatch: expected {tool_use.id}, got {result.tool_use_id}",
            )

        return _truncate_result(result, self.max_output_chars)

    async def adispatch_many(
        self,
        tool_uses: Sequence[ToolUse],
        context: ToolExecutionContext | None = None,
    ) -> tuple[ToolResult, ...]:
        """按并发安全分组执行多个 ToolUse，并保持输入顺序。

        Execute multiple tool uses by concurrency-safe groups while preserving order.
        """

        if not tool_uses:
            return ()

        group_calls: dict[tuple[str, str | int], list[tuple[int, ToolUse]]] = defaultdict(list)
        for index, tool_use in enumerate(tool_uses):
            tool = self._tools.get(tool_use.name)

            if tool is None or tool.concurrency_group is None:
                # 无共享资源的调用各自成组，因此可以并行。
                # Calls without shared resources get separate groups and may run concurrently.
                group_key: tuple[str, str | int] = ("call", index)
            else:
                # 访问同类共享资源的调用进入同一组，按输入顺序串行执行。
                # Calls sharing a resource enter one group and run in input order.
                group_key = ("shared", tool.concurrency_group)

            group_calls[group_key].append((index, tool_use))

        ordered_results: list[ToolResult | None] = [None] * len(tool_uses)

        async def execute_group(items: list[tuple[int, ToolUse]]) -> None:
            for index, tool_use in items:
                ordered_results[index] = await self.dispatch(tool_use, context)

        await asyncio.gather(*(execute_group(items) for items in group_calls.values()))

        if any(result is None for result in ordered_results):
            raise RuntimeError("tool dispatch did not produce every result")
        return tuple(cast(ToolResult, result) for result in ordered_results)

    def dispatch_many(
        self,
        tool_uses: Sequence[ToolUse],
        context: ToolExecutionContext | None = None,
    ) -> tuple[ToolResult, ...]:
        """从同步代码执行多个 ToolUse。

        Execute multiple tool uses from synchronous code.
        """

        return _run_coroutine(self.adispatch_many(tool_uses, context))

    def state_updates(
        self,
        tool_uses: Sequence[ToolUse],
        tool_results: Sequence[ToolResult],
    ) -> dict[str, JsonValue]:
        """按 ToolUse 顺序合并成功执行产生的 Capability State 更新。

        Merge capability-state updates from successful tools in tool-use order.
        """

        if len(tool_uses) != len(tool_results):
            raise ValueError("tool use and result counts must match")

        updates: dict[str, JsonValue] = {}
        for tool_use, tool_result in zip(tool_uses, tool_results, strict=True):
            if tool_result.is_error:
                continue
            tool = self._tools.get(tool_use.name)
            if tool is None or not isinstance(tool, StatefulTool):
                continue
            patch = tool.state_update(tool_use, tool_result)
            updates[patch.namespace] = patch.value
        return updates


__all__ = [
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_TOOL_TIMEOUT_SECONDS",
    "ContextAwareTool",
    "DuplicateToolError",
    "StatefulTool",
    "Tool",
    "ToolDefinition",
    "ToolErrorCode",
    "ToolInput",
    "ToolInputSchema",
    "ToolExecutionContext",
    "ToolRegistry",
    "ToolStatePatch",
]
