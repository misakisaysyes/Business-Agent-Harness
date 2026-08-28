"""MCP 名称、发现隔离、权限和统一 ToolResult 测试。

Tests for MCP naming, discovery isolation, permissions, and unified tool results.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.sessions import Connection
from tests.fakes import FakeSequenceModel

from entrypoints.bootstrap import bootstrap_agent
from harness.agent_loop import get_permission_request
from harness.capabilities.task_system import InMemoryTaskStore
from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider
from harness.permissions import PermissionDecision
from harness.state import AgentState
from harness.tool_use import ToolErrorCode, ToolInput, ToolRegistry
from services.config import AppSettings, MCPServerSettings, MCPSettings, RuntimePathSettings
from services.mcp_tools import (
    LangChainMCPTool,
    MCPIntegration,
    MCPPermissionRule,
    MCPToolAdapter,
    build_mcp_tool_name,
    load_mcp_integration,
    normalize_mcp_name,
)


class FakeRemoteTool:
    """返回预设 ToolMessage 的 LangChain MCP Tool 替身。"""

    def __init__(
        self,
        name: str,
        *,
        read_only: bool = False,
        destructive: bool = False,
        response: str = "remote-ok",
        is_error: bool = False,
        delay: float = 0,
    ) -> None:
        self.name = name
        self.description = f"Remote {name}."
        self.args_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
        self.metadata = {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
        }
        self.response = response
        self.is_error = is_error
        self.delay = delay
        self.calls: list[object] = []

    async def ainvoke(self, input: object) -> object:
        self.calls.append(input)
        await asyncio.sleep(self.delay)
        tool_call = input if isinstance(input, dict) else {}
        return ToolMessage(
            content=self.response,
            tool_call_id=str(tool_call.get("id", "missing")),
            status="error" if self.is_error else "success",
        )


class FakeSource:
    """可按 Server 返回工具、延迟或异常的发现源。"""

    def __init__(
        self,
        tools: dict[str, Sequence[LangChainMCPTool]],
        *,
        failures: dict[str, Exception] | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.tools = tools
        self.failures = failures or {}
        self.delays = delays or {}

    async def discover(
        self,
        server_name: str,
        connection: Connection,
    ) -> Sequence[LangChainMCPTool]:
        await asyncio.sleep(self.delays.get(server_name, 0))
        failure = self.failures.get(server_name)
        if failure is not None:
            raise failure
        return self.tools.get(server_name, ())


class LocalInput(ToolInput):
    text: str


class LocalTool:
    name = "local_echo"
    description = "Echo local input."
    input_schema = LocalInput
    concurrency_group = None

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        value = LocalInput.model_validate(tool_use.input)
        return ToolResult(tool_use_id=tool_use.id, content=value.text)


def mcp_settings(*server_names: str, timeout: float = 1.0) -> MCPSettings:
    return MCPSettings(
        enabled=True,
        discovery_timeout_seconds=timeout,
        servers={
            name: MCPServerSettings(transport="stdio", command="unused")
            for name in server_names
        },
    )


def state() -> AgentState:
    return {"thread_id": "mcp-test", "messages": []}


def test_mcp_names_follow_s19_namespace_and_resolve_normalized_collisions() -> None:
    """特殊字符必须规范化，同名结果必须获得确定性后缀。"""

    assert normalize_mcp_name("docs.v1/search") == "docs_v1_search"
    assert build_mcp_tool_name("docs.v1", "search/doc") == "mcp__docs_v1__search_doc"


async def test_discovery_namespaces_tools_and_uses_annotation_permissions() -> None:
    """readOnly 自动允许，destructive 和未知安全属性请求确认。"""

    read_tool = FakeRemoteTool("search/doc", read_only=True)
    collision_tool = FakeRemoteTool("search?doc", destructive=True)
    unknown_tool = FakeRemoteTool("publish")
    integration = await load_mcp_integration(
        mcp_settings("docs.v1"),
        source=FakeSource(
            {"docs.v1": (read_tool, collision_tool, unknown_tool)}
        ),
    )

    assert tuple(tool.name for tool in integration.tools) == (
        "mcp__docs_v1__search_doc",
        "mcp__docs_v1__search_doc__2",
        "mcp__docs_v1__publish",
    )
    assert "(MCP readOnly)" in integration.tools[0].description
    assert "(MCP destructive)" in integration.tools[1].description

    rule = integration.permission_rules[0]
    decisions = [
        await rule.evaluate(
            ToolUse(id=f"permission-{index}", name=tool.name, input={"query": "x"}),
            state(),
        )
        for index, tool in enumerate(integration.tools)
    ]
    assert [decision.decision for decision in decisions] == [
        PermissionDecision.ALLOW,
        PermissionDecision.ASK,
        PermissionDecision.ASK,
    ]


async def test_mcp_schema_and_errors_use_the_shared_tool_result_contract() -> None:
    """MCP 参数错误和远端错误必须转换成配对的标准 ToolResult。"""

    success_remote = FakeRemoteTool("lookup", read_only=True)
    error_remote = FakeRemoteTool(
        "fail_lookup",
        read_only=True,
        response="remote failed",
        is_error=True,
    )
    success_tool = MCPToolAdapter("docs", "mcp__docs__lookup", success_remote)
    error_tool = MCPToolAdapter("docs", "mcp__docs__fail_lookup", error_remote)
    registry = ToolRegistry((success_tool, error_tool))

    invalid = await registry.dispatch(
        ToolUse(id="invalid", name=success_tool.name, input={})
    )
    succeeded = await registry.dispatch(
        ToolUse(id="success", name=success_tool.name, input={"query": "agent"})
    )
    failed = await registry.dispatch(
        ToolUse(id="failed", name=error_tool.name, input={"query": "agent"})
    )

    assert invalid.is_error
    assert invalid.content["error"] == ToolErrorCode.INVALID_INPUT
    assert success_remote.calls
    assert succeeded == ToolResult(tool_use_id="success", content="remote-ok")
    assert failed == ToolResult(
        tool_use_id="failed",
        content="remote failed",
        is_error=True,
    )


async def test_failed_or_timed_out_server_does_not_remove_local_or_healthy_tools() -> None:
    """单个 Server 失败或超时后，本地工具和健康 Server 必须继续可用。"""

    healthy = FakeRemoteTool("lookup", read_only=True)
    integration = await load_mcp_integration(
        mcp_settings("healthy", "broken", "slow", timeout=0.01),
        source=FakeSource(
            {"healthy": (healthy,)},
            failures={"broken": ConnectionError("offline")},
            delays={"slow": 0.1},
        ),
    )
    registry = ToolRegistry((LocalTool(), *integration.tools))

    local_result = await registry.dispatch(
        ToolUse(id="local", name="local_echo", input={"text": "still works"})
    )

    assert tuple(tool.name for tool in integration.tools) == ("mcp__healthy__lookup",)
    assert {failure.server_name for failure in integration.failures} == {"broken", "slow"}
    assert local_result == ToolResult(tool_use_id="local", content="still works")


async def test_timed_out_mcp_call_does_not_block_parallel_local_tool() -> None:
    """远程调用超时只能失败自身，同批本地 Tool 仍应返回结果。"""

    remote = FakeRemoteTool("slow_lookup", read_only=True, delay=0.1)
    mcp_tool = MCPToolAdapter("slow", "mcp__slow__slow_lookup", remote)
    registry = ToolRegistry((mcp_tool, LocalTool()), timeout_seconds=0.01)

    remote_result, local_result = await registry.adispatch_many(
        (
            ToolUse(
                id="remote-slow",
                name=mcp_tool.name,
                input={"query": "wait"},
            ),
            ToolUse(
                id="local-fast",
                name="local_echo",
                input={"text": "available"},
            ),
        )
    )

    assert remote_result.is_error
    assert remote_result.content["error"] == ToolErrorCode.TIMEOUT
    assert local_result == ToolResult(tool_use_id="local-fast", content="available")


async def test_mcp_permission_rule_passes_through_local_tools() -> None:
    """MCP Rule 不得抢先处理本地 Tool。"""

    rule = MCPPermissionRule(
        (MCPToolAdapter("docs", "mcp__docs__lookup", FakeRemoteTool("lookup")),)
    )

    assert (
        await rule.evaluate(
            ToolUse(id="local", name="local_echo", input={"text": "x"}),
            state(),
        )
        is PermissionDecision.PASSTHROUGH
    )


def test_read_only_mcp_tool_uses_bootstrap_permission_and_hooks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MCP Tool 必须与本地 Tool 共用模型工具池、Permission 和 Hooks。"""

    remote = FakeRemoteTool("lookup", read_only=True)
    tool = MCPToolAdapter("docs", "mcp__docs__lookup", remote)
    integration = MCPIntegration(
        tools=(tool,),
        permission_rules=(MCPPermissionRule((tool,)),),
    )
    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(
                    ToolUse(
                        id="mcp-read-1",
                        name=tool.name,
                        input={"query": "agent"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="done"),
        ]
    )
    settings = AppSettings(
        paths=RuntimePathSettings(workspace_root=tmp_path),
        _env_file=None,
    )
    loop = bootstrap_agent(
        model=cast(ModelProvider, model),
        settings=settings,
        task_store=InMemoryTaskStore(),
        mcp_integration=integration,
    )

    with caplog.at_level(logging.DEBUG, logger="harness.hooks"):
        result = loop.invoke(
            {
                "thread_id": "mcp-bootstrap",
                "messages": [Message(role=MessageRole.USER, content="lookup")],
            }
        )

    assert result["messages"][-1].content == "done"
    assert remote.calls
    assert "calculator" in {tool.name for tool in model.sync_requests[0].tools}
    assert "mcp__docs__lookup" in {tool.name for tool in model.sync_requests[0].tools}
    assert "agent.tool.started" in caplog.text
    assert "agent.tool.finished" in caplog.text


def test_destructive_mcp_tool_pauses_before_remote_call(tmp_path: Path) -> None:
    """destructive MCP Tool 必须在实际发出远程请求前暂停。"""

    remote = FakeRemoteTool("erase", destructive=True)
    tool = MCPToolAdapter("records", "mcp__records__erase", remote)
    integration = MCPIntegration(
        tools=(tool,),
        permission_rules=(MCPPermissionRule((tool,)),),
    )
    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(
                    ToolUse(
                        id="mcp-write-1",
                        name=tool.name,
                        input={"query": "record-1"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="approved"),
        ]
    )
    settings = AppSettings(
        paths=RuntimePathSettings(workspace_root=tmp_path),
        _env_file=None,
    )
    loop = bootstrap_agent(
        model=cast(ModelProvider, model),
        settings=settings,
        task_store=InMemoryTaskStore(),
        mcp_integration=integration,
    )

    paused = loop.invoke(
        {
            "thread_id": "mcp-destructive",
            "messages": [Message(role=MessageRole.USER, content="erase")],
        }
    )
    request = get_permission_request(paused)

    assert request is not None
    assert request.requests[0].tool_name == "mcp__records__erase"
    assert "marked destructive" in request.requests[0].reason
    assert remote.calls == []

    completed = loop.resume("mcp-destructive", True)

    assert completed["messages"][-1].content == "approved"
    assert len(remote.calls) == 1
