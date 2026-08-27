"""Agent Loop 调用和流式输出入口。

Public entry points for invoking and streaming the agent loop.
"""

from collections.abc import Iterator
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from harness.capabilities.context_compact import ContextCompactor
from harness.context import ContextBudget, ContextProvider
from harness.error_recovery import ErrorRecoveryPolicy
from harness.graph import AgentGraph, build_agent_graph
from harness.hooks import Hook, HookFailureMode
from harness.model import ModelProvider
from harness.permissions import PermissionApproval, PermissionRequest, PermissionRule
from harness.state import AgentState
from harness.system_prompt import MemoryProvider, SystemPromptProvider
from harness.tool_use import Tool


class AgentLoop:
    """调用编译后 LangGraph 的最小 Agent Loop 门面。

    Minimal agent-loop facade over a compiled LangGraph.
    """

    def __init__(
        self,
        graph: AgentGraph,
        checkpointer: BaseCheckpointSaver[str],
        tool_names: frozenset[str] = frozenset(),
    ) -> None:
        """保存已经编译完成的 Agent Graph。

        Store the compiled agent graph.
        """

        self.graph = graph
        self.checkpointer = checkpointer
        self.tool_names = tool_names

    def invoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """同步执行一次完整 Graph 并返回最终 State。

        Execute the full graph synchronously and return its final state.
        """

        active_config = config or {"configurable": {"thread_id": state["thread_id"]}}
        return self.graph.invoke(state, config=active_config)

    async def ainvoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """异步执行一次完整 Graph 并返回最终 State。

        Execute the full graph asynchronously and return its final state.
        """

        active_config = config or {"configurable": {"thread_id": state["thread_id"]}}
        return await self.graph.ainvoke(state, config=active_config)

    def stream(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> Iterator[AgentState]:
        """以 State 快照形式流式执行 Graph。

        Stream graph execution as successive state snapshots.
        """

        active_config = config or {"configurable": {"thread_id": state["thread_id"]}}
        return self.graph.stream(state, config=active_config, stream_mode="values")

    def resume(
        self,
        thread_id: str,
        approval: PermissionApproval | bool,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """使用审批结果恢复一个被 interrupt 暂停的 Graph。

        Resume an interrupted graph with a permission approval response.
        """

        active_config = config or {"configurable": {"thread_id": thread_id}}
        resume_value = (
            approval.model_dump(mode="json")
            if isinstance(approval, PermissionApproval)
            else approval
        )
        return self.graph.invoke(Command(resume=resume_value), config=active_config)

    async def aresume(
        self,
        thread_id: str,
        approval: PermissionApproval | bool,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """异步恢复一个被 interrupt 暂停的 Graph。

        Resume an interrupted graph asynchronously with an approval response.
        """

        active_config = config or {"configurable": {"thread_id": thread_id}}
        resume_value = (
            approval.model_dump(mode="json")
            if isinstance(approval, PermissionApproval)
            else approval
        )
        return await self.graph.ainvoke(Command(resume=resume_value), config=active_config)

    async def adelete_thread(self, thread_id: str) -> None:
        """删除指定 Conversation 的全部 Checkpoint。

        Delete every checkpoint associated with a conversation.
        """

        await self.checkpointer.adelete_thread(thread_id)


def get_permission_request(result: AgentState) -> PermissionRequest | None:
    """从被 interrupt 暂停的 Graph 结果中读取审批请求。

    Read a permission request from an interrupted graph result.
    """

    result_values = cast(dict[str, object], result)
    interrupts = result_values.get("__interrupt__")
    if not isinstance(interrupts, (list, tuple)) or not interrupts:
        return None

    interrupt_item = cast(object, interrupts[0])
    value = getattr(interrupt_item, "value", None)
    if value is None:
        return None
    return PermissionRequest.model_validate(value)


def create_agent_loop(
    model: ModelProvider,
    system_prompt: SystemPromptProvider,
    checkpointer: BaseCheckpointSaver[str],
    tools: tuple[Tool, ...] = (),
    permission_rules: tuple[PermissionRule, ...] = (),
    hooks: tuple[Hook, ...] = (),
    hook_failure_mode: HookFailureMode = HookFailureMode.CONTINUE,
    max_iterations: int = 8,
    context_providers: tuple[ContextProvider, ...] = (),
    context_budget: ContextBudget | None = None,
    skill_summaries: tuple[str, ...] = (),
    memory_provider: MemoryProvider | None = None,
    context_compactor: ContextCompactor | None = None,
    error_recovery: ErrorRecoveryPolicy | None = None,
) -> AgentLoop:
    """使用通用依赖创建完整 Agent Loop。

    Create the full agent loop from shared dependencies.
    """

    return AgentLoop(
        graph=build_agent_graph(
            model,
            system_prompt,
            checkpointer,
            tools=tools,
            permission_rules=permission_rules,
            hooks=hooks,
            hook_failure_mode=hook_failure_mode,
            max_iterations=max_iterations,
            context_providers=context_providers,
            context_budget=context_budget,
            skill_summaries=skill_summaries,
            memory_provider=memory_provider,
            context_compactor=context_compactor,
            error_recovery=error_recovery,
        ),
        checkpointer=checkpointer,
        tool_names=frozenset(tool.name for tool in tools),
    )


__all__ = ["AgentLoop", "create_agent_loop", "get_permission_request"]
