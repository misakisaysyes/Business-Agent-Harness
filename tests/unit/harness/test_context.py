"""Context 收集、去重和预算测试。

Context collection, deduplication, and budgeting tests.
"""

from harness.context import (
    CONTEXT_TRUNCATION_SUFFIX,
    ContextBudget,
    ContextFragment,
    ContextManager,
)
from harness.messages import Message, MessageRole
from harness.state import AgentState


class FixedContextProvider:
    """返回固定 Fragment 的 Context Provider 测试替身。"""

    def __init__(self, name: str, fragments: tuple[ContextFragment, ...]) -> None:
        self.name = name
        self.fragments = fragments

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        return self.fragments


def state() -> AgentState:
    return {
        "thread_id": "context-thread",
        "messages": [Message(role=MessageRole.USER, content="current task")],
    }


def test_context_manager_deduplicates_and_uses_stable_priority_order() -> None:
    """相同 key 或内容只保留高优先级项，同优先级保持提供顺序。"""

    provider = FixedContextProvider(
        "fixed",
        (
            ContextFragment(key="low", title="Low", content="duplicate", priority=10),
            ContextFragment(key="first", title="First", content="first", priority=100),
            ContextFragment(key="second", title="Second", content="second", priority=100),
            ContextFragment(key="high", title="High", content="duplicate", priority=200),
        ),
    )

    selection = ContextManager((provider,)).select(state())

    assert tuple(fragment.key for fragment in selection.fragments) == (
        "high",
        "first",
        "second",
    )
    assert selection.omitted_fragments == 1


def test_context_manager_truncates_to_character_budget() -> None:
    """过长的高优先级 Fragment 应截断且总字符数不超过预算。"""

    provider = FixedContextProvider(
        "fixed",
        (
            ContextFragment(
                key="long",
                title="Task",
                content="x" * 200,
                priority=100,
            ),
        ),
    )
    budget = ContextBudget(max_characters=80, max_fragments=1)

    selection = ContextManager((provider,), budget).select(state())

    assert selection.used_characters <= 80
    assert selection.fragments[0].content.endswith(CONTEXT_TRUNCATION_SUFFIX)


def test_context_manager_rejects_duplicate_provider_names() -> None:
    """Provider 名称重复会导致来源不可追踪，因此应拒绝装配。"""

    first = FixedContextProvider("duplicate", ())
    second = FixedContextProvider("duplicate", ())

    try:
        ContextManager((first, second))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate provider names should fail")
