"""真实 stdio Transport 下的 MCP 发现和调用集成测试。

MCP discovery and invocation integration tests over a real stdio transport.
"""

import sys
from pathlib import Path

from harness.messages import ToolUse
from harness.permissions import PermissionDecision, PermissionPipeline
from harness.state import AgentState
from harness.tool_use import ToolRegistry
from services.config import MCPServerSettings, MCPSettings
from services.mcp_tools import load_mcp_integration


async def test_stdio_mock_server_discovers_invokes_and_preserves_mcp_errors() -> None:
    """官方 SDK 应完成握手、tools/list、tools/call 和 isError 转换。"""

    server_path = Path(__file__).parents[1] / "fixtures/mock_mcp_server.py"
    settings = MCPSettings(
        enabled=True,
        discovery_timeout_seconds=10,
        servers={
            "demo.server": MCPServerSettings(
                transport="stdio",
                command=sys.executable,
                args=(str(server_path),),
            )
        },
    )

    integration = await load_mcp_integration(settings)

    assert integration.failures == ()
    assert tuple(tool.name for tool in integration.tools) == (
        "mcp__demo_server__lookup",
        "mcp__demo_server__erase",
        "mcp__demo_server__fail_lookup",
    )

    registry = ToolRegistry(integration.tools, timeout_seconds=10)
    lookup = await registry.dispatch(
        ToolUse(
            id="lookup-1",
            name="mcp__demo_server__lookup",
            input={"query": "MCP"},
        )
    )
    failed = await registry.dispatch(
        ToolUse(
            id="failed-1",
            name="mcp__demo_server__fail_lookup",
            input={"query": "MCP"},
        )
    )

    assert not lookup.is_error
    assert "mock-result:MCP" in str(lookup.content)
    assert failed.is_error
    assert "mock remote failure:MCP" in str(failed.content)

    state: AgentState = {"thread_id": "mcp-integration", "messages": []}
    permissions = PermissionPipeline(
        integration.permission_rules,
        (tool.name for tool in integration.tools),
    )
    read_decision = await permissions.evaluate(
        ToolUse(
            id="permission-read",
            name="mcp__demo_server__lookup",
            input={"query": "MCP"},
        ),
        state,
    )
    destructive_decision = await permissions.evaluate(
        ToolUse(
            id="permission-write",
            name="mcp__demo_server__erase",
            input={"record_id": "record-1"},
        ),
        state,
    )

    assert read_decision.decision is PermissionDecision.ALLOW
    assert destructive_decision.decision is PermissionDecision.ASK
