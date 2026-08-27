"""受授权目录限制的本地文件读取 Tool。

Local file reader restricted to authorized directories.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from harness.messages import ToolResult, ToolUse
from harness.tool_use import ToolInput


class FileReaderInput(ToolInput):
    """File Reader Tool 的参数。

    Input for the file-reader tool.
    """

    path: str = Field(min_length=1, max_length=1_024)
    max_chars: int = Field(default=20_000, ge=1, le=100_000)


class FileReaderTool:
    """只允许读取配置根目录内 UTF-8 文本的 Tool。

    Read UTF-8 text files only from configured root directories.
    """

    name = "file_reader"
    description = (
        "Read a UTF-8 text file from the configured workspace. "
        "Some paths may require user approval."
    )
    input_schema = FileReaderInput
    concurrency_group = "filesystem"

    def __init__(
        self,
        allowed_roots: Sequence[str | Path],
        default_root: str | Path | None = None,
    ) -> None:
        if not allowed_roots:
            raise ValueError("at least one allowed root is required")
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)
        self.default_root = Path(default_root).resolve() if default_root is not None else None
        if self.default_root is not None and not any(
            self.default_root.is_relative_to(root) for root in self.allowed_roots
        ):
            raise ValueError("default root must be inside an allowed root")

    def resolve(self, requested_path: str) -> Path:
        """解析授权范围内的文件路径。

        Resolve a file path within the authorized roots.
        """

        requested = Path(requested_path)
        if requested.is_absolute():
            candidate = requested.resolve()
            if any(candidate.is_relative_to(root) for root in self.allowed_roots):
                return candidate
            raise PermissionError("file path is outside the authorized roots")

        search_roots = (
            (self.default_root, *self.allowed_roots)
            if self.default_root is not None
            else self.allowed_roots
        )
        authorized_candidates: list[Path] = []
        for root in search_roots:
            candidate = (root / requested).resolve()
            if any(candidate.is_relative_to(allowed_root) for allowed_root in self.allowed_roots):
                authorized_candidates.append(candidate)
                if candidate.exists():
                    return candidate

        if authorized_candidates:
            return authorized_candidates[0]
        raise PermissionError("file path is outside the authorized roots")

    def _read(self, tool_input: FileReaderInput) -> str:
        target = self.resolve(tool_input.path)
        if not target.is_file():
            raise FileNotFoundError(f"file does not exist: {tool_input.path}")

        content = target.read_text(encoding="utf-8")
        if len(content) <= tool_input.max_chars:
            return content
        return content[: tool_input.max_chars] + "... [truncated by file_reader]"

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """读取文件并返回与 ToolUse 配对的内容。

        Read the file and return content paired with the tool use.
        """

        tool_input = FileReaderInput.model_validate(tool_use.input)
        content = await asyncio.to_thread(self._read, tool_input)
        return ToolResult(tool_use_id=tool_use.id, content=content)


__all__ = ["FileReaderInput", "FileReaderTool"]
