"""供 MCP 集成测试使用的本地 stdio Server。

Local stdio server used by MCP integration tests.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("mock-mcp-server")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def lookup(query: str) -> str:
    """Return a deterministic lookup result."""

    return f"mock-result:{query}"


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def erase(record_id: str) -> str:
    """Pretend to erase a record without performing a real side effect."""

    return f"mock-erased:{record_id}"


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def fail_lookup(query: str) -> str:
    """Return an MCP protocol-level tool error."""

    raise RuntimeError(f"mock remote failure:{query}")


if __name__ == "__main__":
    server.run(transport="stdio")
