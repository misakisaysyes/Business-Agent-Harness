"""具体的 Model Provider 适配器。

Concrete model-provider adapters.
"""

import json
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import JsonValue

from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider, ModelRequest
from services.config import ModelSettings


class ModelConfigurationError(ValueError):
    """模型配置缺失或包含不支持的值。

    Raised when model configuration is missing or unsupported.
    """


def _to_langchain_messages(request: ModelRequest) -> list[BaseMessage]:
    """把通用 ModelRequest 转换成 LangChain Message。

    Convert a shared ModelRequest into LangChain messages.
    """

    messages: list[BaseMessage] = [SystemMessage(content=request.system_prompt)]

    for message in request.messages:
        if message.role is MessageRole.USER:
            messages.append(HumanMessage(content=message.content))
        elif message.role is MessageRole.ASSISTANT:
            tool_calls = [
                ToolCall(
                    name=tool_use.name,
                    args=tool_use.input,
                    id=tool_use.id,
                    type="tool_call",
                )
                for tool_use in message.tool_uses
            ]
            additional_kwargs: dict[str, JsonValue] = {}
            reasoning_content = message.provider_metadata.get("reasoning_content")
            if isinstance(reasoning_content, str):
                additional_kwargs["reasoning_content"] = reasoning_content
            messages.append(
                AIMessage(
                    content=message.content,
                    tool_calls=tool_calls,
                    additional_kwargs=additional_kwargs,
                )
            )
        elif message.role is MessageRole.SYSTEM:
            messages.append(SystemMessage(content=message.content))
        elif message.role is MessageRole.TOOL:
            for result in message.tool_results:
                content = (
                    result.content
                    if isinstance(result.content, str)
                    else json.dumps(result.content, ensure_ascii=False)
                )
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=result.tool_use_id,
                        status="error" if result.is_error else "success",
                    )
                )
        else:
            raise ValueError(f"unsupported message role: {message.role}")

    return messages


def _to_agent_message(response: BaseMessage) -> Message:
    """把 LangChain 模型响应转换成通用 Assistant Message。

    Convert a LangChain model response into a shared assistant message.
    """

    if not isinstance(response, AIMessage):
        raise ValueError("chat model must return an AIMessage")

    content = str(response.text)
    tool_uses: list[ToolUse] = []
    for tool_call in response.tool_calls:
        tool_call_id = tool_call["id"]
        if tool_call_id is None:
            raise ValueError("chat model returned a tool call without an ID")
        tool_uses.append(
            ToolUse(
                id=tool_call_id,
                name=tool_call["name"],
                input=cast(dict[str, JsonValue], tool_call["args"]),
            )
        )
    if not content and not tool_uses:
        raise ValueError("chat model returned an empty response")

    provider_metadata: dict[str, JsonValue] = {}
    reasoning_content = response.additional_kwargs.get("reasoning_content")
    if isinstance(reasoning_content, str):
        provider_metadata["reasoning_content"] = reasoning_content
    if response.usage_metadata is not None:
        usage = response.usage_metadata
        provider_metadata["usage"] = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    finish_reason = response.response_metadata.get("finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = response.response_metadata.get("stop_reason")
    if isinstance(finish_reason, str):
        provider_metadata["finish_reason"] = finish_reason

    return Message(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_uses=tuple(tool_uses),
        provider_metadata=provider_metadata,
    )


class _PreservingChatOpenAI(ChatOpenAI):
    """保留 OpenAI 兼容服务返回的思考历史字段。

    Preserve thinking-history fields returned by OpenAI-compatible providers.

    Kimi K3 requires ``reasoning_content`` to be sent back unchanged during
    multi-turn conversations and tool use.  The generic ChatOpenAI adapter
    intentionally drops provider-specific response fields, so this small
    compatibility adapter restores the required round trip.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        payload = cast(
            dict[str, Any],
            super()._get_request_payload(  # pyright: ignore[reportUnknownMemberType]
                input_, stop=stop, **kwargs
            ),
        )
        serialized_messages = payload.get("messages")
        if not isinstance(serialized_messages, list):
            return payload
        serialized_message_items = cast(list[Any], serialized_messages)

        for source, serialized in zip(messages, serialized_message_items, strict=False):
            if not isinstance(source, AIMessage) or not isinstance(serialized, dict):
                continue
            reasoning_content = source.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning_content, str):
                serialized["reasoning_content"] = reasoning_content
        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        response_data = cast(
            dict[str, Any],
            response if isinstance(response, dict) else response.model_dump(),
        )
        result = super()._create_chat_result(  # pyright: ignore[reportUnknownMemberType]
            response, generation_info
        )

        choices = response_data.get("choices")
        if not isinstance(choices, list):
            return result
        choice_items = cast(list[Any], choices)
        for generation, choice in zip(result.generations, choice_items, strict=False):
            message = generation.message
            choice_data = cast(dict[str, Any], choice) if isinstance(choice, dict) else {}
            raw_message = choice_data.get("message")
            if not isinstance(raw_message, dict):
                continue
            raw_message_data = cast(dict[str, Any], raw_message)
            reasoning_content = raw_message_data.get("reasoning_content")
            if isinstance(message, AIMessage) and isinstance(reasoning_content, str):
                message.additional_kwargs["reasoning_content"] = reasoning_content
        return result


def _model_for_request(
    client: BaseChatModel,
    request: ModelRequest,
) -> BaseChatModel | Runnable[LanguageModelInput, AIMessage]:
    """按当前请求绑定 Tool Schema。

    Bind tool schemas for the current model request.
    """

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in request.tools
    ]
    model: BaseChatModel | Runnable[LanguageModelInput, AIMessage] = client
    if request.tools and request.required_tool is None:
        model = client.bind_tools(tools)
    elif request.tools:
        # Kimi K3 始终开启 Thinking，目前不接受指定函数对象形式的 tool_choice。
        # 只暴露目标工具再使用 required，既能保持确定性，也兼容 K3 Thinking。
        # Kimi K3 always enables thinking and currently rejects a named-function
        # tool_choice. Expose only the target tool and require one call instead.
        required_tools = [
            tool for tool in tools if tool["function"]["name"] == request.required_tool
        ]
        if len(required_tools) != 1:
            raise ValueError(f"required tool schema is not unique: {request.required_tool}")
        model = client.bind_tools(required_tools, tool_choice="required")

    if request.max_output_tokens is not None:
        model = model.bind(max_tokens=request.max_output_tokens)
    return model


class AnthropicModelProvider:
    """使用 LangChain ChatAnthropic 实现 ModelProvider。

    ModelProvider implementation backed by LangChain ChatAnthropic.
    """

    name = "anthropic"

    def __init__(self, settings: ModelSettings) -> None:
        if settings.model_id is None or settings.api_key is None:
            raise ModelConfigurationError("Anthropic model ID and API key are required")

        self._client: BaseChatModel = ChatAnthropic(
            model_name=settings.model_id,
            api_key=settings.api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=0,
            stop=None,
        )

    def invoke(self, request: ModelRequest) -> Message:
        """同步调用 ChatAnthropic。

        Invoke ChatAnthropic synchronously.
        """

        response = _model_for_request(self._client, request).invoke(_to_langchain_messages(request))
        return _to_agent_message(response)

    async def ainvoke(self, request: ModelRequest) -> Message:
        """异步调用 ChatAnthropic。

        Invoke ChatAnthropic asynchronously.
        """

        response = await _model_for_request(self._client, request).ainvoke(
            _to_langchain_messages(request)
        )
        return _to_agent_message(response)


class _OpenAICompatibleModelProvider:
    """OpenAI 兼容接口的共享 ModelProvider 实现。

    Shared ModelProvider implementation for OpenAI-compatible APIs.
    """

    name = "openai_compatible"
    service_name = "OpenAI-compatible"

    def __init__(self, settings: ModelSettings) -> None:
        if settings.model_id is None or settings.api_key is None:
            raise ModelConfigurationError(f"{self.service_name} model ID and API key are required")
        if settings.base_url is None:
            raise ModelConfigurationError(f"{self.service_name} base URL is required")

        self._client: BaseChatModel = _PreservingChatOpenAI(
            model=settings.model_id,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def invoke(self, request: ModelRequest) -> Message:
        """同步调用 OpenAI 兼容的 Chat Completions API。

        Invoke an OpenAI-compatible Chat Completions API synchronously.
        """

        response = _model_for_request(self._client, request).invoke(_to_langchain_messages(request))
        return _to_agent_message(response)

    async def ainvoke(self, request: ModelRequest) -> Message:
        """异步调用 OpenAI 兼容的 Chat Completions API。

        Invoke an OpenAI-compatible Chat Completions API asynchronously.
        """

        response = await _model_for_request(self._client, request).ainvoke(
            _to_langchain_messages(request)
        )
        return _to_agent_message(response)


class MoonshotModelProvider(_OpenAICompatibleModelProvider):
    """使用 Moonshot OpenAI 兼容接口实现 ModelProvider。

    ModelProvider implementation backed by Moonshot's OpenAI-compatible API.
    """

    name = "moonshot"
    service_name = "Moonshot"


class DeepSeekModelProvider(_OpenAICompatibleModelProvider):
    """使用 DeepSeek OpenAI 兼容接口实现 ModelProvider。

    ModelProvider implementation backed by DeepSeek's OpenAI-compatible API.
    """

    name = "deepseek"
    service_name = "DeepSeek"


def create_model_provider(settings: ModelSettings) -> ModelProvider:
    """根据外部配置创建具体 ModelProvider。

    Create a concrete ModelProvider from external configuration.
    """

    if settings.provider is None:
        raise ModelConfigurationError("AGENT_MODEL__PROVIDER is required")
    if settings.model_id is None:
        raise ModelConfigurationError("AGENT_MODEL__MODEL_ID is required")
    if settings.api_key is None:
        raise ModelConfigurationError("AGENT_MODEL__API_KEY is required")

    provider_name = settings.provider.lower()
    if provider_name == "anthropic":
        return AnthropicModelProvider(settings)
    if provider_name in {"moonshot", "kimi"}:
        return MoonshotModelProvider(settings)
    if provider_name in {"deepseek", "ds"}:
        return DeepSeekModelProvider(settings)

    raise ModelConfigurationError(f"unsupported model provider: {settings.provider}")


__all__ = [
    "AnthropicModelProvider",
    "DeepSeekModelProvider",
    "ModelConfigurationError",
    "MoonshotModelProvider",
    "create_model_provider",
]
