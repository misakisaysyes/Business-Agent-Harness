"""AgentProfile 契约与能力装配。

AgentProfile contracts and capability composition.
"""

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from harness.context import ContextProvider
from harness.hooks import Hook, HookFailureMode
from harness.permissions import PermissionRule
from harness.system_prompt import MemoryProvider, SystemPromptProvider
from harness.tool_use import Tool


class ModelConfigRef(BaseModel):
    """由 Bootstrap 解析的模型配置引用。

    Reference to model configuration resolved by bootstrap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)


class Capability(BaseModel):
    """可插拔 Harness 能力及其无业务实现的配置。

    Pluggable harness capability and its implementation-independent options.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    options: dict[str, JsonValue] = Field(default_factory=dict)


class AgentProfile(BaseModel):
    """描述一个业务 Agent 所需组件的不可变装配契约。

    Immutable composition contract describing the components required by a business agent.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    model: ModelConfigRef
    system_prompt: SystemPromptProvider
    tools: tuple[Tool, ...] = ()
    permission_rules: tuple[PermissionRule, ...] = ()
    hooks: tuple[Hook, ...] = ()
    hook_failure_mode: HookFailureMode = HookFailureMode.CONTINUE
    context_providers: tuple[ContextProvider, ...] = ()
    skill_summaries: tuple[str, ...] = ()
    memory_provider: MemoryProvider | None = None
    capabilities: tuple[Capability, ...] = ()


__all__ = ["AgentProfile", "Capability", "ModelConfigRef"]
