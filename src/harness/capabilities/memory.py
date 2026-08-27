"""跨会话 Memory 契约、选择器和工具。

Cross-session memory contracts, selector, and tools.
"""

import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.messages import MessageRole, ToolResult, ToolUse
from harness.permissions import PermissionDecision, PermissionResult
from harness.state import AgentState
from harness.tool_use import ToolInput

MEMORY_INDEX_FILE = "MEMORY.md"
MEMORY_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
DEFAULT_MAX_MEMORIES = 200
DEFAULT_MAX_SELECTED_MEMORIES = 5
DEFAULT_MAX_MEMORY_CONTENT_CHARACTERS = 4_096
DEFAULT_MAX_MEMORY_INDEX_CHARACTERS = 25_000


class MemoryType(StrEnum):
    """与 learn-claude-code s09 对齐的四类长期 Memory。"""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class MemoryDraft(BaseModel):
    """一次新建或合并更新 Memory 的规范输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=MEMORY_NAME_PATTERN)
    memory_type: MemoryType
    description: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=10_000)
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    source: str = Field(min_length=1, max_length=500)


class MemoryEntry(BaseModel):
    """当前有效的完整 Memory 内容和版本元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=MEMORY_NAME_PATTERN)
    memory_type: MemoryType
    description: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=10_000)
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    source: str = Field(min_length=1, max_length=500)
    created_at: datetime
    updated_at: datetime
    revision: int = Field(ge=1)


class MemoryIndexEntry(BaseModel):
    """常驻 Memory 索引使用的低成本元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    memory_type: MemoryType
    description: str
    tags: tuple[str, ...] = ()
    updated_at: datetime
    revision: int = Field(ge=1)

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> "MemoryIndexEntry":
        return cls(
            name=entry.name,
            memory_type=entry.memory_type,
            description=entry.description,
            tags=entry.tags,
            updated_at=entry.updated_at,
            revision=entry.revision,
        )


@runtime_checkable
class MemoryStore(Protocol):
    """Harness 使用的用户级 Memory Store 协议。"""

    def upsert(self, draft: MemoryDraft, tool_use_id: str) -> MemoryEntry:
        """按稳定名称新建或合并更新 Memory，并保留审计记录。"""
        ...

    def get(self, name: str) -> MemoryEntry | None:
        """读取一个完整 Memory。"""
        ...

    def list_index(self) -> tuple[MemoryIndexEntry, ...]:
        """读取适合进入 Prompt 的 Memory 索引。"""
        ...

    def search(
        self,
        query: str,
        memory_types: Sequence[MemoryType] = (),
        limit: int = DEFAULT_MAX_SELECTED_MEMORIES,
    ) -> tuple[MemoryEntry, ...]:
        """选择与查询相关的少量完整 Memory。"""
        ...


class MemorySelectionConfig(BaseModel):
    """每个模型请求的 Memory 索引和正文预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_selected_memories: int = Field(default=DEFAULT_MAX_SELECTED_MEMORIES, ge=1, le=20)
    max_memory_content_characters: int = Field(
        default=DEFAULT_MAX_MEMORY_CONTENT_CHARACTERS,
        ge=100,
    )
    max_memory_index_characters: int = Field(
        default=DEFAULT_MAX_MEMORY_INDEX_CHARACTERS,
        ge=500,
    )


class MemoryPromptProvider:
    """把 Memory 索引和相关正文选择性注入 System Prompt。"""

    def __init__(
        self,
        store: MemoryStore,
        config: MemorySelectionConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or MemorySelectionConfig()

    def provide(self, state: AgentState) -> tuple[str, ...]:
        """返回低成本索引和与当前用户请求相关的正文。"""

        index = self.store.list_index()
        if not index:
            return ()

        index_lines = [
            f"[{item.memory_type.value}] {item.name}: {item.description}"
            for item in index
        ]
        rendered_index = "Available cross-session memory index:\n" + "\n".join(index_lines)
        rendered_index = rendered_index[: self.config.max_memory_index_characters]

        query = self._latest_user_text(state)
        selected = self.store.search(
            query,
            limit=self.config.max_selected_memories,
        )
        entries = [rendered_index]
        for memory in selected:
            content = memory.content[: self.config.max_memory_content_characters]
            entries.append(
                f"Relevant memory [{memory.memory_type.value}] {memory.name} "
                f"(revision {memory.revision}):\n{content}"
            )
        return tuple(entries)

    @staticmethod
    def _latest_user_text(state: AgentState) -> str:
        for message in reversed(state["messages"]):
            if message.role is MessageRole.USER:
                return message.content
        return ""


class MemoryWriteInput(ToolInput):
    """Memory Write Tool 输入；同名写入采用合并更新语义。"""

    name: str = Field(min_length=1, max_length=64, pattern=MEMORY_NAME_PATTERN)
    memory_type: MemoryType
    description: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=10_000)
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    source: str = Field(min_length=1, max_length=500)


class MemorySearchInput(ToolInput):
    """Memory Search Tool 输入。"""

    query: str = Field(min_length=1, max_length=2_000)
    memory_types: tuple[MemoryType, ...] = ()
    limit: int = Field(default=DEFAULT_MAX_SELECTED_MEMORIES, ge=1, le=20)


class MemoryWriteTool:
    """显式写入跨会话 Memory；同名条目保留版本审计并更新当前值。"""

    name = "memory_write"
    description = (
        "Persist information useful across conversations. Use only when the user explicitly "
        "asks to remember something or confirms a stable preference, feedback, project fact, "
        "or reference. Reusing a name updates that memory and preserves its audit history."
    )
    input_schema = MemoryWriteInput
    concurrency_group = "memory_store"

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        validated = MemoryWriteInput.model_validate(tool_use.input)
        draft = MemoryDraft.model_validate(validated.model_dump(mode="json"))
        entry = await asyncio.to_thread(self.store.upsert, draft, tool_use.id)
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "name": entry.name,
                "memory_type": entry.memory_type.value,
                "revision": entry.revision,
                "updated": entry.revision > 1,
            },
        )


class MemorySearchTool:
    """按当前用户 Store 显式检索少量相关 Memory。"""

    name = "memory_search"
    description = (
        "Search the current user's cross-conversation memories. Use when the memory index "
        "suggests relevant information but the needed detail is not already in the prompt."
    )
    input_schema = MemorySearchInput
    concurrency_group = "memory_store"

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        validated = MemorySearchInput.model_validate(tool_use.input)
        entries = await asyncio.to_thread(
            self.store.search,
            validated.query,
            validated.memory_types,
            validated.limit,
        )
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "memories": [entry.model_dump(mode="json") for entry in entries],
            },
        )


class MemoryWritePermissionRule:
    """跨会话持久化前始终请求用户确认。"""

    name = "confirm_memory_write"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != MemoryWriteTool.name:
            return PermissionDecision.PASSTHROUGH
        memory_name = tool_use.input.get("name", "unknown")
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"cross-conversation memory persistence requires approval: {memory_name}",
        )


class MemorySearchPermissionRule:
    """允许读取已经按当前用户隔离的 Memory Store。"""

    name = "allow_memory_search"

    async def evaluate(
        self,
        tool_use: ToolUse,
        state: AgentState,
    ) -> PermissionResult | PermissionDecision:
        if tool_use.name != MemorySearchTool.name:
            return PermissionDecision.PASSTHROUGH
        return PermissionResult(
            decision=PermissionDecision.ALLOW,
            reason="memory_search reads only the current user's isolated memory store",
        )


def memory_search_terms(text: str) -> frozenset[str]:
    """提取英文词和中文二元组，供无额外模型调用的本地选择器使用。"""

    normalized = text.casefold()
    words = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_-]+", normalized)
        if word not in {"the", "and", "for", "with", "what", "this", "that"}
    }
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_pairs = {
        run[index : index + 2]
        for run in chinese_runs
        for index in range(max(0, len(run) - 1))
    }
    return frozenset(words | chinese_pairs)


__all__ = [
    "DEFAULT_MAX_MEMORIES",
    "DEFAULT_MAX_MEMORY_CONTENT_CHARACTERS",
    "DEFAULT_MAX_MEMORY_INDEX_CHARACTERS",
    "DEFAULT_MAX_SELECTED_MEMORIES",
    "MEMORY_INDEX_FILE",
    "MEMORY_NAME_PATTERN",
    "MemoryDraft",
    "MemoryEntry",
    "MemoryIndexEntry",
    "MemoryPromptProvider",
    "MemorySearchInput",
    "MemorySearchPermissionRule",
    "MemorySearchTool",
    "MemorySelectionConfig",
    "MemoryStore",
    "MemoryType",
    "MemoryWriteInput",
    "MemoryWritePermissionRule",
    "MemoryWriteTool",
    "memory_search_terms",
]
