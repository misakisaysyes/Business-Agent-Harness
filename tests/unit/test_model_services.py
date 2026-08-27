"""具体模型服务工厂测试。

Tests for the concrete model-service factory.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from pydantic import SecretStr

from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider, ModelRequest
from harness.tool_use import ToolDefinition
from services.config import ModelSettings
from services.models import (
    DeepSeekModelProvider,
    ModelConfigurationError,
    MoonshotModelProvider,
    _model_for_request,
    _PreservingChatOpenAI,
    _to_agent_message,
    _to_langchain_messages,
    create_model_provider,
)


def test_model_factory_requires_external_configuration() -> None:
    """缺少模型配置时工厂应在调用 API 前失败。

    The factory should fail before any API call when model configuration is missing.
    """

    with pytest.raises(ModelConfigurationError, match="PROVIDER"):
        create_model_provider(ModelSettings())


def test_model_factory_rejects_unknown_provider() -> None:
    """未知模型提供方应产生明确错误。

    An unknown model provider should produce an explicit error.
    """

    settings = ModelSettings(
        provider="unknown",
        model_id="test-model",
        api_key=SecretStr("test-key"),
    )

    with pytest.raises(ModelConfigurationError, match="unsupported model provider"):
        create_model_provider(settings)


def test_anthropic_factory_returns_model_provider_without_network_call() -> None:
    """完整 Anthropic 配置应创建 Provider，但不得在构造时访问网络。

    Complete Anthropic configuration should create a provider without a construction-time call.
    """

    settings = ModelSettings(
        provider="anthropic",
        model_id="test-model",
        api_key=SecretStr("test-key"),
    )

    provider = create_model_provider(settings)

    assert isinstance(provider, ModelProvider)
    assert provider.name == "anthropic"


def test_moonshot_factory_returns_model_provider_without_network_call() -> None:
    """完整 Moonshot 配置应创建 Provider，但不得在构造时访问网络。

    Complete Moonshot configuration should create a provider without a construction-time call.
    """

    settings = ModelSettings(
        provider="moonshot",
        model_id="kimi-k3",
        api_key=SecretStr("test-key"),
        base_url="https://api.moonshot.cn/v1",
        temperature=1,
    )

    provider = create_model_provider(settings)

    assert isinstance(provider, MoonshotModelProvider)
    assert isinstance(provider, ModelProvider)
    assert provider.name == "moonshot"


def test_moonshot_factory_requires_base_url() -> None:
    """Moonshot 配置缺少 Base URL 时应产生明确错误。

    Moonshot configuration should fail clearly when its base URL is missing.
    """

    settings = ModelSettings(
        provider="moonshot",
        model_id="kimi-k3",
        api_key=SecretStr("test-key"),
        temperature=1,
    )

    with pytest.raises(ModelConfigurationError, match="base URL"):
        create_model_provider(settings)


def test_deepseek_factory_returns_model_provider_without_network_call() -> None:
    """完整 DeepSeek 配置应创建 OpenAI 兼容 Provider。

    Complete DeepSeek settings should create an OpenAI-compatible provider.
    """

    settings = ModelSettings(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        api_key=SecretStr("test-key"),
        base_url="https://api.deepseek.com",
    )

    provider = create_model_provider(settings)

    assert isinstance(provider, DeepSeekModelProvider)
    assert isinstance(provider, ModelProvider)
    assert provider.name == "deepseek"


def test_internal_tool_messages_convert_to_langchain_messages() -> None:
    """内部 ToolUse 和 ToolResult 应转换成 LangChain 标准消息。

    Internal tool uses and results should convert to standard LangChain messages.
    """

    tool_use = ToolUse(id="tool-1", name="calculator", input={"expression": "2 + 3"})
    request = ModelRequest(
        system_prompt="Use tools when needed.",
        messages=(
            Message(role=MessageRole.USER, content="计算 2 + 3"),
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(
                role=MessageRole.TOOL,
                tool_results=(ToolResult(tool_use_id=tool_use.id, content=5),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="calculator",
                description="Calculate arithmetic.",
                parameters={"type": "object"},
            ),
        ),
    )

    messages = _to_langchain_messages(request)

    assert isinstance(messages[-2], AIMessage)
    assert messages[-2].tool_calls[0]["id"] == tool_use.id
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == tool_use.id
    assert messages[-1].content == "5"


def test_langchain_tool_call_converts_to_internal_assistant_message() -> None:
    """LangChain Tool Call 应转换成内部 ToolUse。

    A LangChain tool call should convert to an internal tool use.
    """

    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 3"},
                "id": "tool-1",
                "type": "tool_call",
            }
        ],
    )

    message = _to_agent_message(response)

    assert message.role is MessageRole.ASSISTANT
    assert message.tool_uses == (
        ToolUse(id="tool-1", name="calculator", input={"expression": "2 + 3"}),
    )


def test_langchain_usage_metadata_is_normalized_for_accounting() -> None:
    """Provider 返回的标准 Token 用量应保留在内部元数据中。"""

    response = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 3,
            "total_tokens": 14,
        },
    )

    message = _to_agent_message(response)

    assert message.provider_metadata["usage"] == {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
    }


def test_finish_reason_and_output_limit_are_preserved_for_recovery() -> None:
    """适配器应保留停止原因，并把请求级输出上限传给 LangChain。"""

    message = _to_agent_message(
        AIMessage(content="partial", response_metadata={"finish_reason": "length"})
    )
    model = _PreservingChatOpenAI(model="kimi-k3", api_key="test-key")
    bound = _model_for_request(
        model,
        ModelRequest(
            system_prompt="Answer.",
            messages=(Message(role=MessageRole.USER, content="test"),),
            max_output_tokens=8_192,
        ),
    )

    assert message.provider_metadata["finish_reason"] == "length"
    assert isinstance(bound, RunnableBinding)
    assert bound.kwargs["max_tokens"] == 8_192

    anthropic_message = _to_agent_message(
        AIMessage(content="partial", response_metadata={"stop_reason": "max_tokens"})
    )
    assert anthropic_message.provider_metadata["finish_reason"] == "max_tokens"


def test_kimi_reasoning_content_survives_message_round_trip() -> None:
    """Kimi K3 的思考历史应在内部消息和下一次请求之间保留。"""

    response = AIMessage(
        content="answer",
        additional_kwargs={"reasoning_content": "private thinking state"},
    )

    internal = _to_agent_message(response)
    converted = _to_langchain_messages(
        ModelRequest(
            system_prompt="Use tools when needed.",
            messages=(internal,),
        )
    )

    assert internal.provider_metadata == {"reasoning_content": "private thinking state"}
    assert isinstance(converted[-1], AIMessage)
    assert converted[-1].additional_kwargs["reasoning_content"] == "private thinking state"


def test_kimi_compatible_chat_model_preserves_reasoning_in_payload() -> None:
    """兼容适配器应从响应提取 reasoning_content 并在下轮原样发送。"""

    model = _PreservingChatOpenAI(model="kimi-k3", api_key="test-key")
    result = model._create_chat_result(
        {
            "model": "kimi-k3",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "private thinking state",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )

    assistant = result.generations[0].message
    assert isinstance(assistant, AIMessage)
    assert assistant.additional_kwargs["reasoning_content"] == "private thinking state"

    payload = model._get_request_payload([assistant])
    assert payload["messages"][0]["reasoning_content"] == "private thinking state"


def test_required_tool_is_the_only_schema_and_uses_required_tool_choice() -> None:
    """强制调用应只暴露目标工具，并使用 Kimi Thinking 兼容的 required。"""

    model = _PreservingChatOpenAI(model="kimi-k3", api_key="test-key")
    request = ModelRequest(
        system_prompt="Use the required tool.",
        messages=(Message(role=MessageRole.USER, content="write"),),
        tools=(
            ToolDefinition(
                name="calculator",
                description="Calculate.",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="report_writer",
                description="Write a report.",
                parameters={"type": "object", "properties": {}},
            ),
        ),
        required_tool="report_writer",
    )

    bound = _model_for_request(model, request)

    assert isinstance(bound, RunnableBinding)
    assert bound.kwargs["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in bound.kwargs["tools"]] == [
        "report_writer"
    ]
