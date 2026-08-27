"""报告生成和 Artifact 写入 Tool。

Report generation and artifact-writing tool.
"""

import asyncio

from pydantic import Field

from harness.messages import ToolResult, ToolUse
from harness.tool_use import ToolInput
from services.artifacts import ArtifactStore


class ReportWriterInput(ToolInput):
    """Report Writer Tool 的参数。

    Input for the report-writer tool.
    """

    path: str = Field(min_length=1, max_length=1_024)
    content: str = Field(min_length=1, max_length=1_000_000)
    overwrite: bool = False


class ReportWriterTool:
    """把报告写入受限 ArtifactStore 的 Tool。

    Write reports into a restricted artifact store.
    """

    name = "report_writer"
    description = "Write a UTF-8 report into the artifact directory. Overwrite requires approval."
    input_schema = ReportWriterInput
    concurrency_group = "filesystem"

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """写入报告并返回 Artifact 相对路径。

        Write a report and return its relative artifact path.
        """

        tool_input = ReportWriterInput.model_validate(tool_use.input)
        write = await asyncio.to_thread(
            self.store.write_text,
            tool_input.path,
            tool_input.content,
            tool_input.overwrite,
        )
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "path": str(write.path.relative_to(self.store.root)),
                "overwritten": write.overwritten,
            },
        )


__all__ = ["ReportWriterInput", "ReportWriterTool"]
