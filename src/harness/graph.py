"""LangGraph 节点、边和条件路由。

LangGraph nodes, edges, and conditional routing.
"""

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, cast

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command, Overwrite, interrupt
from pydantic import JsonValue

from harness.capabilities.context_compact import (
    CONTEXT_COMPACT_NAMESPACE,
    CompactionResult,
    ContextCompactor,
)
from harness.context import ContextBudget, ContextManager, ContextProvider
from harness.error_recovery import (
    ErrorRecoveryPolicy,
    OutputTokenRecoveryError,
    PromptTooLongRecoveryError,
    is_output_truncated,
    is_prompt_too_long_error,
    next_output_token_limit,
)
from harness.hooks import (
    Hook,
    HookFailureMode,
    HookRegistry,
    PostToolUse,
    PreToolUse,
    Stop,
    UserPromptSubmit,
)
from harness.logging import AgentLog
from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider, ModelRequest
from harness.permissions import (
    PermissionApproval,
    PermissionDecision,
    PermissionPipeline,
    PermissionRequest,
    PermissionRequestItem,
    PermissionResult,
    PermissionRule,
)
from harness.state import AgentState, AgentStateUpdate, AgentStopReason
from harness.system_prompt import MemoryProvider, SystemPromptBuilder, SystemPromptProvider
from harness.tool_use import Tool, ToolDefinition, ToolExecutionContext, ToolRegistry

PREPARE_NODE = "prepare"
COMPACT_NODE = "compact"
MODEL_NODE = "model"
PERMISSION_NODE = "permission"
TOOL_NODE = "tool"
FINAL_NODE = "final"

MODEL_ROUTE = "model"
PERMISSION_ROUTE = "permission"
FINAL_ROUTE = "final"
DEFAULT_MAX_ITERATIONS = 8
log = AgentLog(__name__)


def _state_log_fields(state: AgentState) -> dict[str, object]:
    metadata = state.get("metadata", {})
    current_trace_id = log.context_fields().get("trace_id")
    return {
        "thread_id": state["thread_id"],
        "run_id": metadata.get("run_id"),
        "trace_id": (
            current_trace_id
            if isinstance(current_trace_id, str)
            else metadata.get("trace_id")
        ),
        "iteration_count": state.get("iteration_count", 0),
        "message_count": len(state["messages"]),
    }


class AgentGraph(Protocol):
    """AgentLoop 使用的编译 Graph 最小接口。

    Minimal compiled-graph interface consumed by AgentLoop.
    """

    def invoke(
        self,
        state: AgentState | Command[str],
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """同步执行 Graph。

        Invoke the graph synchronously.
        """
        ...

    async def ainvoke(
        self,
        state: AgentState | Command[str],
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """异步执行 Graph。

        Invoke the graph asynchronously.
        """
        ...

    def stream(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
        stream_mode: Literal["values"] = "values",
    ) -> Iterator[AgentState]:
        """以完整 State 快照流式执行 Graph。

        Stream graph execution as full state snapshots.
        """
        ...


def _create_model_request(
    state: AgentState,
    system_prompt: str,
    tool_definitions: tuple[ToolDefinition, ...],
    max_output_tokens: int | None = None,
) -> ModelRequest:
    """根据当前 State 创建一次标准模型请求。

    Create one normalized model request from the current state.
    """

    required_tool: str | None = None
    if state.get("iteration_count", 0) == 0:
        configured_tool = state.get("metadata", {}).get("required_tool")
        if configured_tool is not None:
            if not isinstance(configured_tool, str):
                raise ValueError("required_tool metadata must be a string")
            required_tool = configured_tool

    return ModelRequest(
        system_prompt=system_prompt,
        messages=tuple(state["messages"]),
        tools=tool_definitions,
        max_output_tokens=max_output_tokens,
        required_tool=required_tool,
    )


def _create_model_update(response: Message) -> AgentStateUpdate:
    """校验模型响应并创建增量状态。

    Validate a model response and create a partial state update.
    """

    if response.role is not MessageRole.ASSISTANT:
        raise ValueError("model provider must return an assistant message")
    return {"messages": [response]}


def _latest_assistant_message(state: AgentState) -> Message:
    """返回 Graph 刚追加的 Assistant Message。

    Return the assistant message most recently appended by the graph.
    """

    message = state["messages"][-1]
    if message.role is not MessageRole.ASSISTANT:
        raise ValueError("latest message must be an assistant message")
    return message


def _latest_user_message(state: AgentState) -> Message:
    """返回当前 Agent Turn 最近的一条 User Message。

    Return the most recent user message for the current agent turn.
    """

    for message in reversed(state["messages"]):
        if message.role is MessageRole.USER:
            return message
    raise ValueError("agent state requires at least one user message")


def build_agent_graph(
    model: ModelProvider,
    system_prompt: SystemPromptProvider,
    checkpointer: BaseCheckpointSaver[str],
    tools: Sequence[Tool] = (),
    permission_rules: Sequence[PermissionRule] = (),
    hooks: Sequence[Hook] = (),
    hook_failure_mode: HookFailureMode = HookFailureMode.CONTINUE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    context_providers: Sequence[ContextProvider] = (),
    context_budget: ContextBudget | None = None,
    skill_summaries: Sequence[str] = (),
    memory_provider: MemoryProvider | None = None,
    context_compactor: ContextCompactor | None = None,
    error_recovery: ErrorRecoveryPolicy | None = None,
) -> AgentGraph:
    """构建包含模型、权限、工具和终止路由的 Agent Graph。

    Build an agent graph with model, permission, tool, and termination routing.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    recovery_policy = error_recovery or ErrorRecoveryPolicy()
    tool_registry = ToolRegistry(
        tools,
        timeout_seconds=recovery_policy.tool_timeout_seconds,
    )
    tool_definitions = tool_registry.definitions()
    permission_pipeline = PermissionPipeline(permission_rules, tool_registry.names())
    hook_registry = HookRegistry(hooks, failure_mode=hook_failure_mode)
    prompt_builder = SystemPromptBuilder(
        system_prompt,
        ContextManager(context_providers, context_budget),
        skill_summaries=skill_summaries,
        memory_provider=memory_provider,
    )

    def invoke_with_output_recovery(request: ModelRequest) -> Message:
        """同步提高一次输出上限，仍截断时安全终止。"""

        active_request = request
        response = model.invoke(active_request)
        retries = 0
        while is_output_truncated(response):
            current_limit = (
                active_request.max_output_tokens
                or recovery_policy.initial_max_output_tokens
            )
            next_limit = next_output_token_limit(current_limit, recovery_policy)
            if retries >= recovery_policy.max_output_retries or next_limit <= current_limit:
                raise OutputTokenRecoveryError(
                    "model output remained truncated after bounded recovery"
                )
            retries += 1
            log.record(
                "agent.model.output_limit_retry",
                retry_number=retries,
                max_retries=recovery_policy.max_output_retries,
                max_output_tokens=next_limit,
            )
            active_request = active_request.model_copy(
                update={"max_output_tokens": next_limit}
            )
            response = model.invoke(active_request)
        return response

    async def ainvoke_with_output_recovery(request: ModelRequest) -> Message:
        """异步提高一次输出上限，仍截断时安全终止。"""

        active_request = request
        response = await model.ainvoke(active_request)
        retries = 0
        while is_output_truncated(response):
            current_limit = (
                active_request.max_output_tokens
                or recovery_policy.initial_max_output_tokens
            )
            next_limit = next_output_token_limit(current_limit, recovery_policy)
            if retries >= recovery_policy.max_output_retries or next_limit <= current_limit:
                raise OutputTokenRecoveryError(
                    "model output remained truncated after bounded recovery"
                )
            retries += 1
            log.record(
                "agent.model.output_limit_retry",
                retry_number=retries,
                max_retries=recovery_policy.max_output_retries,
                max_output_tokens=next_limit,
            )
            active_request = active_request.model_copy(
                update={"max_output_tokens": next_limit}
            )
            response = await model.ainvoke(active_request)
        return response

    def merge_capability_state(
        state: AgentState,
        namespace: str,
        value: JsonValue,
    ) -> dict[str, JsonValue]:
        capability_state = dict(state.get("capability_state", {}))
        capability_state[namespace] = value
        return capability_state

    def compaction_update(
        state: AgentState,
        result: CompactionResult,
        response: Message | None = None,
    ) -> AgentStateUpdate:
        """使用 Overwrite 真正替换活跃消息，而不是触发追加 Reducer。"""

        if not result.changed and response is None:
            return {}
        messages = [*result.messages]
        if response is not None:
            messages.append(response)
        update = {
            "messages": Overwrite(value=messages),
            "capability_state": merge_capability_state(
                state,
                CONTEXT_COMPACT_NAMESPACE,
                result.state.model_dump(mode="json"),
            ),
        }
        return cast(AgentStateUpdate, update)

    def prepare_turn(state: AgentState) -> AgentStateUpdate:
        if state.get("cancel_requested", False):
            return {
                "iteration_count": 0,
                "stop_reason": AgentStopReason.CANCELLED,
            }

        hook_result = hook_registry.dispatch_sync(
            UserPromptSubmit(state=state, message=_latest_user_message(state))
        )
        log.record(
            "agent.turn.prepared",
            **_state_log_fields(state),
            status="blocked" if hook_result.blocked else "ready",
        )
        return {
            "iteration_count": 0,
            "stop_reason": AgentStopReason.HOOK_BLOCKED if hook_result.blocked else None,
        }

    async def aprepare_turn(state: AgentState) -> AgentStateUpdate:
        if state.get("cancel_requested", False):
            return {
                "iteration_count": 0,
                "stop_reason": AgentStopReason.CANCELLED,
            }

        hook_result = await hook_registry.dispatch(
            UserPromptSubmit(state=state, message=_latest_user_message(state))
        )
        log.record(
            "agent.turn.prepared",
            **_state_log_fields(state),
            status="blocked" if hook_result.blocked else "ready",
        )
        return {
            "iteration_count": 0,
            "stop_reason": AgentStopReason.HOOK_BLOCKED if hook_result.blocked else None,
        }

    def compact_context(state: AgentState) -> AgentStateUpdate:
        if context_compactor is None or not context_compactor.config.enabled:
            return {}
        with log.operation("agent.context.compact", **_state_log_fields(state)) as outcome:
            result = context_compactor.compact(state)
            outcome["message_count"] = len(result.messages)
            return compaction_update(state, result)

    async def acompact_context(state: AgentState) -> AgentStateUpdate:
        if context_compactor is None or not context_compactor.config.enabled:
            return {}
        with log.operation("agent.context.compact", **_state_log_fields(state)) as outcome:
            result = await context_compactor.acompact(state)
            outcome["message_count"] = len(result.messages)
            return compaction_update(state, result)

    def reactive_model_update(
        state: AgentState,
        iteration_count: int,
    ) -> AgentStateUpdate:
        if context_compactor is None or context_compactor.config.max_reactive_retries == 0:
            raise RuntimeError("reactive compaction is disabled")
        result = context_compactor.compact(state, reason="reactive", force_summary=True)
        compacted_state = cast(
            AgentState,
            {
                **state,
                "messages": list(result.messages),
                "capability_state": merge_capability_state(
                    state,
                    CONTEXT_COMPACT_NAMESPACE,
                    result.state.model_dump(mode="json"),
                ),
            },
        )
        prompt = prompt_builder.build(compacted_state, tool_definitions)
        response = invoke_with_output_recovery(
            _create_model_request(
                compacted_state,
                prompt,
                tool_definitions,
                recovery_policy.initial_max_output_tokens,
            )
        )
        _create_model_update(response)
        update = compaction_update(state, result, response)
        update["iteration_count"] = iteration_count + 1
        return update

    async def areactive_model_update(
        state: AgentState,
        iteration_count: int,
    ) -> AgentStateUpdate:
        if context_compactor is None or context_compactor.config.max_reactive_retries == 0:
            raise RuntimeError("reactive compaction is disabled")
        result = await context_compactor.acompact(
            state,
            reason="reactive",
            force_summary=True,
        )
        compacted_state = cast(
            AgentState,
            {
                **state,
                "messages": list(result.messages),
                "capability_state": merge_capability_state(
                    state,
                    CONTEXT_COMPACT_NAMESPACE,
                    result.state.model_dump(mode="json"),
                ),
            },
        )
        prompt = prompt_builder.build(compacted_state, tool_definitions)
        response = await ainvoke_with_output_recovery(
            _create_model_request(
                compacted_state,
                prompt,
                tool_definitions,
                recovery_policy.initial_max_output_tokens,
            )
        )
        _create_model_update(response)
        update = compaction_update(state, result, response)
        update["iteration_count"] = iteration_count + 1
        return update

    def invoke_model(state: AgentState) -> AgentStateUpdate:
        if state.get("cancel_requested", False):
            return {"stop_reason": AgentStopReason.CANCELLED}

        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= max_iterations:
            return {"stop_reason": AgentStopReason.MAX_ITERATIONS}

        with log.operation(
            "agent.model.invoke",
            **_state_log_fields(state),
            max_iterations=max_iterations,
            tool_definition_count=len(tool_definitions),
            max_output_tokens=recovery_policy.initial_max_output_tokens,
        ) as outcome:
            prompt = prompt_builder.build(state, tool_definitions)
            request = _create_model_request(
                state,
                prompt,
                tool_definitions,
                recovery_policy.initial_max_output_tokens,
            )
            try:
                response = invoke_with_output_recovery(request)
                update = _create_model_update(response)
            except Exception as error:
                if not is_prompt_too_long_error(error):
                    raise
                if (
                    context_compactor is None
                    or context_compactor.config.max_reactive_retries == 0
                ):
                    raise PromptTooLongRecoveryError(
                        "prompt exceeds the model context window and reactive compact is disabled"
                    ) from error
                try:
                    update = reactive_model_update(state, iteration_count)
                    outcome["status"] = "recovered_after_compaction"
                    return update
                except Exception as retry_error:
                    if is_prompt_too_long_error(retry_error):
                        raise PromptTooLongRecoveryError(
                            "prompt still exceeds the model context window after reactive compact"
                        ) from retry_error
                    raise
            outcome["response_tool_use_count"] = len(response.tool_uses)
            outcome["response_has_content"] = bool(response.content)
            update["iteration_count"] = iteration_count + 1
            return update

    async def ainvoke_model(state: AgentState) -> AgentStateUpdate:
        if state.get("cancel_requested", False):
            return {"stop_reason": AgentStopReason.CANCELLED}

        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= max_iterations:
            return {"stop_reason": AgentStopReason.MAX_ITERATIONS}

        with log.operation(
            "agent.model.invoke",
            **_state_log_fields(state),
            max_iterations=max_iterations,
            tool_definition_count=len(tool_definitions),
            max_output_tokens=recovery_policy.initial_max_output_tokens,
        ) as outcome:
            prompt = prompt_builder.build(state, tool_definitions)
            request = _create_model_request(
                state,
                prompt,
                tool_definitions,
                recovery_policy.initial_max_output_tokens,
            )
            try:
                response = await ainvoke_with_output_recovery(request)
                update = _create_model_update(response)
            except Exception as error:
                if not is_prompt_too_long_error(error):
                    raise
                if (
                    context_compactor is None
                    or context_compactor.config.max_reactive_retries == 0
                ):
                    raise PromptTooLongRecoveryError(
                        "prompt exceeds the model context window and reactive compact is disabled"
                    ) from error
                try:
                    update = await areactive_model_update(state, iteration_count)
                    outcome["status"] = "recovered_after_compaction"
                    return update
                except Exception as retry_error:
                    if is_prompt_too_long_error(retry_error):
                        raise PromptTooLongRecoveryError(
                            "prompt still exceeds the model context window after reactive compact"
                        ) from retry_error
                    raise
            outcome["response_tool_use_count"] = len(response.tool_uses)
            outcome["response_has_content"] = bool(response.content)
            update["iteration_count"] = iteration_count + 1
            return update

    def resolve_permission_results(
        tool_uses: tuple[ToolUse, ...],
        permission_results: tuple[PermissionResult, ...],
    ) -> tuple[tuple[ToolUse, PermissionResult, bool], ...]:
        """处理中断审批并返回每个 ToolUse 的最终授权状态。

        Resolve interrupt approvals and return final authorization for each tool use.
        """

        if len(tool_uses) != len(permission_results):
            raise RuntimeError("permission result count does not match tool use count")

        ask_items = tuple(
            PermissionRequestItem(
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                input=tool_use.input,
                reason=result.reason,
            )
            for tool_use, result in zip(tool_uses, permission_results, strict=True)
            if result.decision is PermissionDecision.ASK
        )

        approval: PermissionApproval | None = None
        if ask_items:
            request = PermissionRequest(requests=ask_items)
            approval = PermissionApproval.from_resume(interrupt(request.model_dump(mode="json")))

        resolved: list[tuple[ToolUse, PermissionResult, bool]] = []
        for tool_use, result in zip(tool_uses, permission_results, strict=True):
            is_allowed = result.decision is PermissionDecision.ALLOW
            if result.decision is PermissionDecision.ASK:
                if approval is None:
                    raise RuntimeError("ASK permission result requires an approval")
                is_allowed = approval.is_approved(tool_use.id)

            resolved.append((tool_use, result, is_allowed))

        return tuple(resolved)

    def denied_tool_result(
        tool_use: ToolUse,
        reason: str,
        error_code: str,
    ) -> ToolResult:
        """创建被 Permission 或 PreToolUse Hook 阻断的结果。

        Create a result blocked by permission or a PreToolUse hook.
        """

        return ToolResult(
            tool_use_id=tool_use.id,
            content={"error": error_code, "reason": reason},
            is_error=True,
        )

    def create_permission_update(
        state: AgentState,
        tool_uses: tuple[ToolUse, ...],
        permission_results: tuple[PermissionResult, ...],
    ) -> AgentStateUpdate:
        """同步触发 PreToolUse Hooks 并生成执行计划。

        Dispatch PreToolUse hooks synchronously and create the execution plan.
        """

        resolved_permissions = resolve_permission_results(tool_uses, permission_results)
        allowed: list[ToolUse] = []
        denied: list[ToolResult] = []
        for tool_use, result, permission_granted in resolved_permissions:
            hook_result = hook_registry.dispatch_sync(
                PreToolUse(
                    state=state,
                    tool_use=tool_use,
                    permission_result=result,
                    permission_granted=permission_granted,
                )
            )
            if permission_granted and not hook_result.blocked:
                allowed.append(tool_use)
            else:
                denied.append(
                    denied_tool_result(
                        tool_use,
                        hook_result.reason or result.reason,
                        "hook_blocked" if permission_granted else "permission_denied",
                    )
                )

        return {
            "pending_tool_uses": allowed,
            "pending_tool_results": denied,
            "capability_state": permission_history_update(state, resolved_permissions),
        }

    async def acreate_permission_update(
        state: AgentState,
        tool_uses: tuple[ToolUse, ...],
        permission_results: tuple[PermissionResult, ...],
    ) -> AgentStateUpdate:
        """异步触发 PreToolUse Hooks 并生成执行计划。

        Dispatch PreToolUse hooks asynchronously and create the execution plan.
        """

        resolved_permissions = resolve_permission_results(tool_uses, permission_results)
        allowed: list[ToolUse] = []
        denied: list[ToolResult] = []
        for tool_use, result, permission_granted in resolved_permissions:
            hook_result = await hook_registry.dispatch(
                PreToolUse(
                    state=state,
                    tool_use=tool_use,
                    permission_result=result,
                    permission_granted=permission_granted,
                )
            )
            if permission_granted and not hook_result.blocked:
                allowed.append(tool_use)
            else:
                denied.append(
                    denied_tool_result(
                        tool_use,
                        hook_result.reason or result.reason,
                        "hook_blocked" if permission_granted else "permission_denied",
                    )
                )

        return {
            "pending_tool_uses": allowed,
            "pending_tool_results": denied,
            "capability_state": permission_history_update(state, resolved_permissions),
        }

    def permission_history_update(
        state: AgentState,
        resolved: tuple[tuple[ToolUse, PermissionResult, bool], ...],
    ) -> dict[str, JsonValue]:
        capability_state = dict(state.get("capability_state", {}))
        raw_history = capability_state.get("permission_history", [])
        history = list(raw_history) if isinstance(raw_history, list) else []
        history.extend(
            {
                "tool_use_id": tool_use.id,
                "tool_name": tool_use.name,
                "decision": result.decision.value,
                "allowed": permission_granted,
                "reason": result.reason,
            }
            for tool_use, result, permission_granted in resolved
        )
        capability_state["permission_history"] = cast(JsonValue, history[-100:])
        return capability_state

    def check_permissions(state: AgentState) -> AgentStateUpdate:
        """同步评估 ToolUse 权限，并在 ASK 时暂停 Graph。

        Evaluate tool permissions synchronously and pause the graph for ASK.
        """

        tool_uses = _latest_assistant_message(state).tool_uses
        if not tool_uses:
            raise ValueError("permission node requires at least one tool use")
        results = permission_pipeline.evaluate_many_sync(tool_uses, state)
        log.record(
            "agent.permission.evaluated",
            **_state_log_fields(state),
            tool_count=len(tool_uses),
            permission_allow_count=sum(
                result.decision is PermissionDecision.ALLOW for result in results
            ),
            permission_ask_count=sum(
                result.decision is PermissionDecision.ASK for result in results
            ),
            permission_deny_count=sum(
                result.decision is PermissionDecision.DENY for result in results
            ),
        )
        return create_permission_update(state, tool_uses, results)

    async def acheck_permissions(state: AgentState) -> AgentStateUpdate:
        """异步评估 ToolUse 权限，并在 ASK 时暂停 Graph。

        Evaluate tool permissions asynchronously and pause the graph for ASK.
        """

        tool_uses = _latest_assistant_message(state).tool_uses
        if not tool_uses:
            raise ValueError("permission node requires at least one tool use")
        results = await permission_pipeline.evaluate_many(tool_uses, state)
        log.record(
            "agent.permission.evaluated",
            **_state_log_fields(state),
            tool_count=len(tool_uses),
            permission_allow_count=sum(
                result.decision is PermissionDecision.ALLOW for result in results
            ),
            permission_ask_count=sum(
                result.decision is PermissionDecision.ASK for result in results
            ),
            permission_deny_count=sum(
                result.decision is PermissionDecision.DENY for result in results
            ),
        )
        return await acreate_permission_update(state, tool_uses, results)

    def execute_tools(state: AgentState) -> AgentStateUpdate:
        tool_uses = tuple(state.get("pending_tool_uses", ()))
        context = ToolExecutionContext(
            thread_id=state["thread_id"],
            metadata=dict(state.get("metadata", {})),
        )
        with log.operation(
            "agent.tools.execute",
            **_state_log_fields(state),
            tool_count=len(tool_uses),
        ) as outcome:
            executed = tool_registry.dispatch_many(tool_uses, context)
            for tool_use, tool_result in zip(tool_uses, executed, strict=True):
                hook_registry.dispatch_sync(
                    PostToolUse(state=state, tool_use=tool_use, tool_result=tool_result)
                )
            outcome["failed_count"] = sum(result.is_error for result in executed)
            return create_tool_update(state, tool_uses, executed)

    async def aexecute_tools(state: AgentState) -> AgentStateUpdate:
        tool_uses = tuple(state.get("pending_tool_uses", ()))
        context = ToolExecutionContext(
            thread_id=state["thread_id"],
            metadata=dict(state.get("metadata", {})),
        )
        with log.operation(
            "agent.tools.execute",
            **_state_log_fields(state),
            tool_count=len(tool_uses),
        ) as outcome:
            executed = await tool_registry.adispatch_many(tool_uses, context)
            for tool_use, tool_result in zip(tool_uses, executed, strict=True):
                await hook_registry.dispatch(
                    PostToolUse(state=state, tool_use=tool_use, tool_result=tool_result)
                )
            outcome["failed_count"] = sum(result.is_error for result in executed)
            return create_tool_update(state, tool_uses, executed)

    def create_tool_update(
        state: AgentState,
        executed_tool_uses: tuple[ToolUse, ...],
        executed: tuple[ToolResult, ...],
    ) -> AgentStateUpdate:
        """合并已执行和被拒绝的结果，并恢复原 ToolUse 顺序。

        Merge executed and denied results while restoring original tool-use order.
        """

        original_tool_uses = _latest_assistant_message(state).tool_uses
        all_results = [*state.get("pending_tool_results", ()), *executed]
        results_by_id = {result.tool_use_id: result for result in all_results}

        try:
            ordered_results = tuple(results_by_id[tool_use.id] for tool_use in original_tool_uses)
        except KeyError as error:
            raise RuntimeError(f"missing tool result for tool use: {error.args[0]}") from error

        update: AgentStateUpdate = {
            "messages": [Message(role=MessageRole.TOOL, tool_results=ordered_results)],
            "pending_tool_uses": [],
            "pending_tool_results": [],
        }
        capability_updates = tool_registry.state_updates(executed_tool_uses, executed)
        if capability_updates:
            capability_state = dict(state.get("capability_state", {}))
            capability_state.update(capability_updates)
            update["capability_state"] = capability_state
        return update

    def finalize(state: AgentState) -> AgentStateUpdate:
        stop_reason = state.get("stop_reason")
        if stop_reason is None:
            last_message = _latest_assistant_message(state)
            if last_message.tool_uses and state.get("iteration_count", 0) >= max_iterations:
                stop_reason = AgentStopReason.MAX_ITERATIONS
            else:
                stop_reason = AgentStopReason.COMPLETED
        hook_registry.dispatch_sync(Stop(state=state, reason=stop_reason))
        log.record(
            "agent.turn.finished",
            **_state_log_fields(state),
            stop_reason=stop_reason.value,
            max_iterations=max_iterations,
        )
        return {"stop_reason": stop_reason}

    async def afinalize(state: AgentState) -> AgentStateUpdate:
        stop_reason = state.get("stop_reason")
        if stop_reason is None:
            last_message = _latest_assistant_message(state)
            if last_message.tool_uses and state.get("iteration_count", 0) >= max_iterations:
                stop_reason = AgentStopReason.MAX_ITERATIONS
            else:
                stop_reason = AgentStopReason.COMPLETED
        await hook_registry.dispatch(Stop(state=state, reason=stop_reason))
        log.record(
            "agent.turn.finished",
            **_state_log_fields(state),
            stop_reason=stop_reason.value,
            max_iterations=max_iterations,
        )
        return {"stop_reason": stop_reason}

    def route_after_prepare(state: AgentState) -> Literal["model", "final"]:
        if state.get("stop_reason") is not None:
            return FINAL_ROUTE
        return MODEL_ROUTE

    def route_after_model(state: AgentState) -> Literal["permission", "final"]:
        if state.get("stop_reason") is not None:
            return FINAL_ROUTE

        response = _latest_assistant_message(state)
        if not response.tool_uses:
            return FINAL_ROUTE
        if state.get("iteration_count", 0) >= max_iterations:
            return FINAL_ROUTE
        return PERMISSION_ROUTE

    prepare_node = RunnableLambda(
        prepare_turn,
        afunc=aprepare_turn,
        name=PREPARE_NODE,
    )
    compact_node = RunnableLambda(
        compact_context,
        afunc=acompact_context,
        name=COMPACT_NODE,
    )
    model_node = RunnableLambda(
        invoke_model,
        afunc=ainvoke_model,
        name=MODEL_NODE,
    )
    permission_node = RunnableLambda(
        check_permissions,
        afunc=acheck_permissions,
        name=PERMISSION_NODE,
    )
    tool_node = RunnableLambda(
        execute_tools,
        afunc=aexecute_tools,
        name=TOOL_NODE,
    )
    final_node = RunnableLambda(
        finalize,
        afunc=afinalize,
        name=FINAL_NODE,
    )

    builder = StateGraph(AgentState)
    builder.add_node(PREPARE_NODE, prepare_node)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node(COMPACT_NODE, compact_node)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node(MODEL_NODE, model_node)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node(PERMISSION_NODE, permission_node)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node(TOOL_NODE, tool_node)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node(FINAL_NODE, final_node)  # pyright: ignore[reportUnknownMemberType]

    builder.add_edge(START, PREPARE_NODE)
    builder.add_conditional_edges(  # pyright: ignore[reportUnknownMemberType]
        PREPARE_NODE,
        route_after_prepare,
        {MODEL_ROUTE: COMPACT_NODE, FINAL_ROUTE: FINAL_NODE},
    )
    builder.add_edge(COMPACT_NODE, MODEL_NODE)
    builder.add_conditional_edges(  # pyright: ignore[reportUnknownMemberType]
        MODEL_NODE,
        route_after_model,
        {PERMISSION_ROUTE: PERMISSION_NODE, FINAL_ROUTE: FINAL_NODE},
    )
    builder.add_edge(PERMISSION_NODE, TOOL_NODE)
    builder.add_edge(TOOL_NODE, COMPACT_NODE)
    builder.add_edge(FINAL_NODE, END)

    compiled = builder.compile(  # pyright: ignore[reportUnknownMemberType]
        checkpointer=checkpointer,
        name="agent_loop",
    )
    return cast(AgentGraph, compiled)


__all__ = [
    "AgentGraph",
    "COMPACT_NODE",
    "DEFAULT_MAX_ITERATIONS",
    "FINAL_NODE",
    "MODEL_NODE",
    "PERMISSION_NODE",
    "PREPARE_NODE",
    "TOOL_NODE",
    "build_agent_graph",
]
