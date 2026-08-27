"""模型请求和 Provider 协议测试。

Tests for model requests and the provider protocol.
"""

import pytest
from pydantic import ValidationError

from harness.messages import Message, MessageRole
from harness.model import ModelProvider, ModelRequest
from harness.tool_use import ToolDefinition


class CompleteModelProvider:
    """同时实现同步和异步接口的测试 Provider。

    Test provider implementing both synchronous and asynchronous interfaces.
    """

    name = "complete"

    def invoke(self, request: ModelRequest) -> Message:
        """返回固定同步响应。

        Return a fixed synchronous response.
        """

        return Message(role=MessageRole.ASSISTANT, content="sync")

    async def ainvoke(self, request: ModelRequest) -> Message:
        """返回固定异步响应。

        Return a fixed asynchronous response.
        """

        return Message(role=MessageRole.ASSISTANT, content="async")


def test_model_request_requires_prompt_and_messages() -> None:
    """模型请求必须同时包含 System Prompt 和会话消息。

    A model request must contain both a system prompt and conversation messages.
    """

    with pytest.raises(ValidationError):
        ModelRequest(system_prompt="", messages=())


def test_runtime_checkable_model_provider_requires_both_interfaces() -> None:
    """完整测试 Provider 应符合运行时 ModelProvider 协议。

    A complete test provider should satisfy the runtime ModelProvider protocol.
    """

    assert isinstance(CompleteModelProvider(), ModelProvider)


def test_required_tool_must_exist_in_request_tools() -> None:
    """不得把不存在的工具发送成强制 tool_choice。"""

    with pytest.raises(ValidationError, match="required tool is not available"):
        ModelRequest(
            system_prompt="Use tools.",
            messages=(Message(role=MessageRole.USER, content="write"),),
            tools=(
                ToolDefinition(
                    name="calculator",
                    description="Calculate.",
                    parameters={"type": "object"},
                ),
            ),
            required_tool="report_writer",
        )
