"""消息、ToolUse 和 ToolResult 契约。

Message, tool-use, and tool-result contracts.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class MessageRole(StrEnum):
    """通用消息角色。

    Shared message roles.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolUse(BaseModel):
    """模型请求执行一次 Tool 的结构化描述。

    Structured description of one tool invocation requested by the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """与 ToolUse ID 对应的执行结果。

    Execution result paired with a ToolUse ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_use_id: str = Field(min_length=1)
    content: JsonValue
    is_error: bool = False


class Message(BaseModel):
    """独立于具体模型提供方的标准消息。

    Model-provider-independent message representation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str = ""
    tool_uses: tuple[ToolUse, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def validate_payload(self) -> "Message":
        """确保消息至少包含文本或一种 Tool 载荷。

        Ensure that a message contains text or at least one tool payload.
        """

        if not self.content and not self.tool_uses and not self.tool_results:
            msg = "message must contain content, tool uses, or tool results"
            raise ValueError(msg)
        return self


__all__ = ["Message", "MessageRole", "ToolResult", "ToolUse"]
