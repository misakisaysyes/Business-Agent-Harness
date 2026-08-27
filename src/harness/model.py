"""Harness 使用的模型提供方协议。

Model provider protocols used by the harness.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.messages import Message
from harness.tool_use import ToolDefinition


class ModelRequest(BaseModel):
    """独立于具体模型 SDK 的标准模型请求。

    Model-SDK-independent request passed to a model provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = ()
    max_output_tokens: int | None = Field(default=None, ge=1)
    required_tool: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$",
    )

    @model_validator(mode="after")
    def required_tool_must_be_available(self) -> "ModelRequest":
        """强制工具必须存在于当前请求的 Tool Schema 中。

        A forced tool must exist in the tool schemas supplied with this request.
        """

        if self.required_tool is None:
            return self
        if self.required_tool not in {tool.name for tool in self.tools}:
            raise ValueError(f"required tool is not available: {self.required_tool}")
        return self


@runtime_checkable
class ModelProvider(Protocol):
    """Model Node 使用的同步和异步模型调用协议。

    Synchronous and asynchronous model invocation protocol used by the model node.
    """

    @property
    def name(self) -> str:
        """返回 Model Provider 的稳定名称。

        Return the stable model-provider name.
        """
        ...

    def invoke(self, request: ModelRequest) -> Message:
        """同步调用模型并返回标准 Assistant Message。

        Invoke the model synchronously and return a normalized assistant message.
        """
        ...

    async def ainvoke(self, request: ModelRequest) -> Message:
        """异步调用模型并返回标准 Assistant Message。

        Invoke the model asynchronously and return a normalized assistant message.
        """
        ...


__all__ = ["ModelProvider", "ModelRequest"]
