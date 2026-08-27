"""通用 Agent State 和 Reducer 测试。

Tests for shared agent state and its reducer.
"""

from harness.messages import Message, MessageRole
from harness.state import append_messages


def test_append_messages_returns_new_list_without_mutating_inputs() -> None:
    """Reducer 应追加消息且不得修改原列表。

    The reducer should append messages without mutating the original lists.
    """

    current = [Message(role=MessageRole.USER, content="问题")]
    updates = [Message(role=MessageRole.ASSISTANT, content="回答")]

    reduced = append_messages(current, updates)

    assert reduced == [*current, *updates]
    assert current == [Message(role=MessageRole.USER, content="问题")]
    assert updates == [Message(role=MessageRole.ASSISTANT, content="回答")]
    assert reduced is not current
    assert reduced is not updates


def test_append_messages_accepts_empty_sides() -> None:
    """Reducer 应支持初始化状态和空更新。

    The reducer should support initial state and empty updates.
    """

    message = Message(role=MessageRole.USER, content="问题")

    assert append_messages(None, [message]) == [message]
    assert append_messages([message], None) == [message]
