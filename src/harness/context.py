"""运行时上下文收集、筛选和预算控制。

Runtime context collection, selection, and budgeting.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.state import AgentState

CONTEXT_TRUNCATION_SUFFIX = "... [truncated]"


class ContextFragment(BaseModel):
    """Context Provider 返回的最小文本片段。

    Minimal text fragment returned by a context provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1)
    priority: int = Field(default=0, ge=0, le=1000)


class ContextBudget(BaseModel):
    """一次模型请求可注入的 Runtime Context 上限。

    Runtime-context limits for one model request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_characters: int = Field(default=8_000, ge=64)
    max_fragments: int = Field(default=16, ge=1)


class ContextSelection(BaseModel):
    """ContextManager 完成筛选后的不可变结果。

    Immutable context selection produced by ContextManager.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragments: tuple[ContextFragment, ...] = ()
    used_characters: int = 0
    omitted_fragments: int = 0


@runtime_checkable
class ContextProvider(Protocol):
    """从 Agent State 生成候选 Context Fragment。

    Produce candidate context fragments from agent state.
    """

    @property
    def name(self) -> str:
        """返回 Context Provider 的唯一名称。

        Return the unique context-provider name.
        """
        ...

    def provide(self, state: AgentState) -> Sequence[ContextFragment]:
        """返回候选上下文；不得返回凭据或未经隔离的内部对象。

        Return candidate context without credentials or unisolated internal objects.
        """
        ...


class ContextManager:
    """收集、去重并按预算选择 Runtime Context。

    Collect, deduplicate, and budget runtime context.
    """

    def __init__(
        self,
        providers: Sequence[ContextProvider] = (),
        budget: ContextBudget | None = None,
    ) -> None:
        provider_names = [provider.name for provider in providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("context provider names must be unique")

        self.providers = tuple(providers)
        self.budget = budget or ContextBudget()

    def select(self, state: AgentState) -> ContextSelection:
        """按优先级稳定选择不超过预算的 Context Fragment。

        Select context fragments by stable priority without exceeding the budget.
        """

        candidates = [
            fragment for provider in self.providers for fragment in provider.provide(state)
        ]
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (-item[1].priority, item[0]),
        )

        selected: list[ContextFragment] = []
        seen_keys: set[str] = set()
        seen_content: set[str] = set()
        used_characters = 0

        for _, fragment in ordered:
            normalized_content = " ".join(fragment.content.split()).casefold()
            if fragment.key in seen_keys or normalized_content in seen_content:
                continue
            if len(selected) >= self.budget.max_fragments:
                continue

            remaining = self.budget.max_characters - used_characters
            title_cost = len(fragment.title)
            content_budget = remaining - title_cost
            if content_budget <= len(CONTEXT_TRUNCATION_SUFFIX):
                continue

            content = fragment.content
            if len(content) > content_budget:
                kept = content_budget - len(CONTEXT_TRUNCATION_SUFFIX)
                content = content[:kept] + CONTEXT_TRUNCATION_SUFFIX

            selected_fragment = fragment.model_copy(update={"content": content})
            selected.append(selected_fragment)
            seen_keys.add(fragment.key)
            seen_content.add(normalized_content)
            used_characters += len(selected_fragment.title) + len(selected_fragment.content)

        return ContextSelection(
            fragments=tuple(selected),
            used_characters=used_characters,
            omitted_fragments=len(candidates) - len(selected),
        )


__all__ = [
    "CONTEXT_TRUNCATION_SUFFIX",
    "ContextBudget",
    "ContextFragment",
    "ContextManager",
    "ContextProvider",
    "ContextSelection",
]
