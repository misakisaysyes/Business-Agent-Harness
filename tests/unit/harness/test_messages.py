"""通用消息契约测试。

Tests for shared message contracts.
"""

import pytest
from pydantic import ValidationError

from harness.messages import Message, MessageRole, ToolResult, ToolUse


def test_text_message_is_valid() -> None:
    """普通文本消息应保留角色和内容。

    A normal text message should preserve its role and content.
    """

    message = Message(role=MessageRole.USER, content="请介绍你能做什么")

    assert message.role is MessageRole.USER
    assert message.content == "请介绍你能做什么"


def test_tool_use_and_result_are_paired_by_identifier() -> None:
    """ToolUse 和 ToolResult 应通过 ID 明确配对。

    ToolUse and ToolResult should be explicitly paired by identifier.
    """

    tool_use = ToolUse(id="tool-1", name="calculator", input={"expression": "1 + 1"})
    tool_result = ToolResult(tool_use_id=tool_use.id, content="2")

    assert tool_result.tool_use_id == tool_use.id
    assert tool_result.is_error is False


def test_empty_message_fails_validation() -> None:
    """不含文本或 Tool 载荷的消息应校验失败。

    A message without text or tool payloads should fail validation.
    """

    with pytest.raises(ValidationError, match="message must contain"):
        Message(role=MessageRole.ASSISTANT)
