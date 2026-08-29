"""文档、文件和报告的业务 Permission 规则。

Business permission rules for documents, files, and reports.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from business.knowledge_assistant.search_routing import is_web_search_tool_name
from business.knowledge_assistant.tools.file_reader import FileReaderInput, FileReaderTool
from business.knowledge_assistant.tools.report_writer import ReportWriterInput
from harness.messages import ToolUse
from harness.permissions import PermissionDecision, PermissionResult
from harness.state import AgentState
from services.artifacts import ArtifactStore, InvalidArtifactPathError


class CalculatorPermissionRule:
    """允许无外部副作用的 Calculator Tool。

    Allow the calculator tool because it has no external side effects.
    """

    name = "allow_calculator"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != "calculator":
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="calculator has no external side effects",
        )


class DocumentSearchPermissionRule:
    """允许已由 Runtime 绑定访问范围的只读知识库检索。"""

    name = "allow_document_search"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != "document_search":
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="document search is read-only and its access scope is runtime-bound",
        )


class DocumentCatalogPermissionRule:
    """允许已由 Runtime 绑定访问范围的只读文档目录查询。"""

    name = "allow_document_catalog"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != "document_catalog":
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="document catalog is read-only and its access scope is runtime-bound",
        )


class SearchModePermissionRule:
    """阻止强制检索模式调用相反来源的搜索 Tool。"""

    name = "search_mode_policy"
    _RAG_TOOLS = frozenset({"document_search", "document_catalog"})

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        mode = state.get("metadata", {}).get("search_mode", "auto")
        if mode == "rag" and is_web_search_tool_name(tool_use.name):
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason="search mode is forced to rag; web search is disabled",
            )
        if mode == "web" and tool_use.name in self._RAG_TOOLS:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason="search mode is forced to web; local RAG search is disabled",
            )
        return PermissionDecision.PASSTHROUGH


class FileReadPermissionRule:
    """只允许 File Reader 读取配置的授权范围。

    Allow the file reader only within its configured authorized roots.
    """

    name = "authorized_file_read"

    def __init__(
        self,
        reader: FileReaderTool,
        auto_allowed_roots: Sequence[str | Path],
    ) -> None:
        if not auto_allowed_roots:
            raise ValueError("at least one auto-allowed root is required")
        self.reader = reader
        self.auto_allowed_roots = tuple(Path(root).resolve() for root in auto_allowed_roots)

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != self.reader.name:
            return PermissionDecision.PASSTHROUGH

        try:
            tool_input = FileReaderInput.model_validate(tool_use.input)
            resolved_path = self.reader.resolve(tool_input.path)
        except (PermissionError, ValidationError) as error:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"file read is outside the authorized scope: {error}",
            )

        if any(resolved_path.is_relative_to(root) for root in self.auto_allowed_roots):
            return PermissionResult(
                decision=PermissionDecision.ALLOW,
                reason="file read is inside an auto-authorized scope",
            )

        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"file read requires one-time approval: {resolved_path}",
        )


class ReportWritePermissionRule:
    """允许新建报告，并在覆盖报告前请求审批。

    Allow new reports and request approval before overwriting reports.
    """

    name = "report_write"

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != "report_writer":
            return PermissionDecision.PASSTHROUGH

        try:
            tool_input = ReportWriterInput.model_validate(tool_use.input)
            self.store.resolve(tool_input.path)
        except (InvalidArtifactPathError, ValidationError) as error:
            return PermissionResult(
                decision=PermissionDecision.DENY,
                reason=f"report path or input is invalid: {error}",
            )

        if tool_input.overwrite:
            return PermissionResult(
                decision=PermissionDecision.ASK,
                reason=f"overwriting report requires approval: {tool_input.path}",
            )

        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason=f"creating a new report is allowed: {tool_input.path}",
        )


class ExternalPublishPermissionRule:
    """默认拒绝对外发布类 Tool。

    Deny external publishing tools by default.
    """

    name = "deny_external_publish"

    def __init__(self, tool_names: Iterable[str] = ("external_publish",)) -> None:
        self.tool_names = frozenset(tool_names)

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name not in self.tool_names:
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.DENY,
            reason=f"external publishing is disabled: {tool_use.name}",
        )


__all__ = [
    "CalculatorPermissionRule",
    "DocumentCatalogPermissionRule",
    "DocumentSearchPermissionRule",
    "ExternalPublishPermissionRule",
    "FileReadPermissionRule",
    "ReportWritePermissionRule",
    "SearchModePermissionRule",
]
