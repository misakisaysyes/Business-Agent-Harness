"""MCP Server 发现、名称隔离、权限和 Harness Tool 适配。

MCP server discovery, namespacing, permissions, and Harness tool adaptation.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from threading import Thread
from typing import Any, Protocol, cast

import structlog
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from pydantic import BaseModel, JsonValue

from harness.messages import ToolResult, ToolUse
from harness.permissions import PermissionDecision, PermissionResult, PermissionRule
from harness.state import AgentState
from harness.tool_use import Tool, ToolInputSchema
from services.config import MCPServerSettings, MCPSettings

MCP_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
MCP_TOOL_PREFIX = "mcp__"

logger = structlog.get_logger(__name__)


class LangChainMCPTool(Protocol):
    """官方适配器返回的 LangChain Tool 最小接口。"""

    name: str
    description: str
    args_schema: object
    metadata: dict[str, Any] | None

    async def ainvoke(self, input: object) -> object: ...


class MCPToolSource(Protocol):
    """允许测试替换真实 MCP 连接的异步发现接口。"""

    async def discover(
        self,
        server_name: str,
        connection: Connection,
    ) -> Sequence[LangChainMCPTool]: ...


class LangChainMCPToolSource:
    """通过 langchain-mcp-adapters 连接并发现一个 Server。"""

    async def discover(
        self,
        server_name: str,
        connection: Connection,
    ) -> Sequence[LangChainMCPTool]:
        client = MultiServerMCPClient(
            {server_name: connection},
            handle_tool_errors=True,
        )
        tools = await client.get_tools(server_name=server_name)
        return cast(Sequence[LangChainMCPTool], tools)


@dataclass(frozen=True, slots=True)
class MCPServerFailure:
    """一个被隔离的 MCP Server 发现错误。"""

    server_name: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class MCPIntegration:
    """Bootstrap 可直接装配的 MCP 工具、权限规则和失败记录。"""

    tools: tuple[Tool, ...] = ()
    permission_rules: tuple[PermissionRule, ...] = ()
    failures: tuple[MCPServerFailure, ...] = ()


class MCPToolAdapter:
    """把一个 LangChain MCP Tool 转换成项目统一 Tool 契约。"""

    concurrency_group = None

    def __init__(
        self,
        server_name: str,
        exposed_name: str,
        remote_tool: LangChainMCPTool,
    ) -> None:
        self.server_name = server_name
        self.remote_name = remote_tool.name
        self.name = exposed_name
        self._remote_tool = remote_tool
        self.input_schema = _tool_input_schema(remote_tool.args_schema)
        metadata = remote_tool.metadata or {}
        self.read_only = _metadata_flag(metadata, "readOnlyHint", "read_only_hint")
        self.destructive = _metadata_flag(
            metadata,
            "destructiveHint",
            "destructive_hint",
        )
        annotation = (
            "readOnly"
            if self.read_only and not self.destructive
            else "destructive"
            if self.destructive
            else "approval required"
        )
        remote_description = remote_tool.description or f"MCP tool {self.remote_name}."
        self.description = f"{remote_description} (MCP {annotation})"

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """调用远程 Tool，并保留 MCP 错误状态和结构化结果。"""

        result = await self._remote_tool.ainvoke(
            {
                "type": "tool_call",
                "id": tool_use.id,
                "name": self.remote_name,
                "args": tool_use.input,
            }
        )
        if isinstance(result, ToolMessage):
            return ToolResult(
                tool_use_id=tool_use.id,
                content=_tool_message_content(result),
                is_error=result.status == "error",
            )
        return ToolResult(tool_use_id=tool_use.id, content=_to_json_value(result))


class MCPPermissionRule:
    """按 MCP Tool annotations 决定自动执行或请求确认。"""

    name = "mcp_tool_annotations"

    def __init__(self, tools: Sequence[MCPToolAdapter]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        tool = self._tools.get(tool_use.name)
        if tool is None:
            return PermissionDecision.PASSTHROUGH
        identity = f"MCP tool {tool.server_name}/{tool.remote_name}"
        if tool.destructive:
            return PermissionResult(
                decision=PermissionDecision.ASK,
                reason=f"{identity} is marked destructive",
            )
        if tool.read_only:
            return PermissionResult(
                decision=PermissionDecision.ALLOW,
                reason=f"{identity} is marked read-only",
            )
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"{identity} has no read-only safety declaration",
        )


def normalize_mcp_name(name: str) -> str:
    """沿用 learn-claude-code s19 的 MCP 名称安全规则。"""

    normalized = MCP_NAME_PATTERN.sub("_", name)
    return normalized or "unnamed"


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """创建 `mcp__server__tool` 命名空间。"""

    return (
        f"{MCP_TOOL_PREFIX}{normalize_mcp_name(server_name)}"
        f"__{normalize_mcp_name(tool_name)}"
    )


async def load_mcp_integration(
    settings: MCPSettings,
    *,
    source: MCPToolSource | None = None,
    reserved_names: Sequence[str] = (),
) -> MCPIntegration:
    """并发发现 Server；单个失败只产生记录，不丢弃健康工具。"""

    if not settings.enabled or not settings.servers:
        return MCPIntegration()

    active_source = source or LangChainMCPToolSource()
    server_items = tuple(settings.servers.items())

    async def discover_one(
        server_name: str,
        server: MCPServerSettings,
    ) -> Sequence[LangChainMCPTool]:
        return await asyncio.wait_for(
            active_source.discover(server_name, _connection_from_settings(server)),
            timeout=settings.discovery_timeout_seconds,
        )

    results = await asyncio.gather(
        *(discover_one(server_name, server) for server_name, server in server_items),
        return_exceptions=True,
    )
    used_names = set(reserved_names)
    tools: list[MCPToolAdapter] = []
    failures: list[MCPServerFailure] = []

    for (server_name, _), result in zip(server_items, results, strict=True):
        if isinstance(result, BaseException):
            failure = MCPServerFailure(
                server_name=server_name,
                error_type=type(result).__name__,
                message=str(result) or type(result).__name__,
            )
            failures.append(failure)
            logger.warning(
                "mcp_server_discovery_failed",
                server_name=server_name,
                error_type=failure.error_type,
            )
            continue

        for remote_tool in result:
            base_name = build_mcp_tool_name(server_name, remote_tool.name)
            exposed_name = _unique_tool_name(base_name, used_names)
            tool = MCPToolAdapter(server_name, exposed_name, remote_tool)
            tools.append(tool)
            used_names.add(exposed_name)

    permission_rules: tuple[PermissionRule, ...] = ()
    if tools:
        permission_rules = (MCPPermissionRule(tools),)
    return MCPIntegration(
        tools=cast(tuple[Tool, ...], tuple(tools)),
        permission_rules=permission_rules,
        failures=tuple(failures),
    )


def load_mcp_integration_sync(
    settings: MCPSettings,
    *,
    source: MCPToolSource | None = None,
    reserved_names: Sequence[str] = (),
) -> MCPIntegration:
    """从同步 Bootstrap 加载 MCP；已有事件循环时放到短生命周期线程。"""

    coroutine = load_mcp_integration(
        settings,
        source=source,
        reserved_names=reserved_names,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    return _run_coroutine_in_thread(coroutine)


def _connection_from_settings(server: MCPServerSettings) -> Connection:
    if server.transport == "stdio":
        connection: dict[str, Any] = {
            "transport": "stdio",
            "command": server.command,
            "args": list(server.args),
        }
        if server.env:
            connection["env"] = {
                key: value.get_secret_value() for key, value in server.env.items()
            }
        return cast(Connection, connection)

    remote_connection: dict[str, Any] = {
        "transport": server.transport,
        "url": server.url,
    }
    if server.headers:
        remote_connection["headers"] = {
            key: value.get_secret_value() for key, value in server.headers.items()
        }
    return cast(Connection, remote_connection)


def _tool_input_schema(value: object) -> ToolInputSchema:
    if isinstance(value, dict):
        return cast(dict[str, JsonValue], value)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value
    raise TypeError("MCP tool args_schema must be JSON Schema or a Pydantic model")


def _metadata_flag(metadata: Mapping[str, Any], *names: str) -> bool:
    return any(metadata.get(name) is True for name in names)


def _unique_tool_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        return base_name
    suffix = 2
    while f"{base_name}__{suffix}" in used_names:
        suffix += 1
    return f"{base_name}__{suffix}"


def _tool_message_content(message: ToolMessage) -> JsonValue:
    content = _to_json_value(message.content)
    if message.artifact is None:
        return content
    return {
        "content": content,
        "artifact": _to_json_value(message.artifact),
    }


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return cast(JsonValue, value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _to_json_value(item) for key, item in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [_to_json_value(item) for item in sequence]
    return str(value)


def _run_coroutine_in_thread(
    coroutine: Coroutine[Any, Any, MCPIntegration],
) -> MCPIntegration:
    results: list[MCPIntegration] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(asyncio.run(coroutine))
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run, name="mcp-discovery", daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not results:
        raise RuntimeError("MCP discovery thread returned no result")
    return results[0]


__all__ = [
    "LangChainMCPToolSource",
    "MCPIntegration",
    "MCPPermissionRule",
    "MCPServerFailure",
    "MCPToolAdapter",
    "MCPToolSource",
    "build_mcp_tool_name",
    "load_mcp_integration",
    "load_mcp_integration_sync",
    "normalize_mcp_name",
]
