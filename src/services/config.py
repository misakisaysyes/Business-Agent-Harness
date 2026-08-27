"""经过校验的应用和环境配置。

Validated application and environment configuration.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """应用部署环境。

    Application deployment environment.
    """

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """支持的日志级别。

    Supported logging levels.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """支持的日志输出格式。

    Supported log output formats.
    """

    CONSOLE = "console"
    JSON = "json"


class ModelSettings(BaseModel):
    """模型提供方的运行配置，不包含任何硬编码凭据。

    Runtime model-provider settings without hard-coded credentials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str | None = Field(default=None, min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_concurrency: int = Field(default=1, ge=1)
    quota_scope: str | None = Field(default=None, min_length=1, max_length=128)


class ModelGatewaySettings(BaseModel):
    """模型路由、重试和跨进程并发配置。

    Model routing, retry, and cross-process concurrency settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_retries: int = Field(default=2, ge=0, le=5)
    rate_limit_max_retries: int = Field(default=1, ge=0, le=5)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0.0)
    retry_max_delay_seconds: float = Field(default=16.0, ge=0.0)
    retry_jitter_seconds: float = Field(default=0.5, ge=0.0)
    lock_directory: Path = Path("/tmp/extensible-agent-template-model-locks")


class ErrorRecoverySettings(BaseModel):
    """模型输出恢复和 Tool 超时的有界配置。

    Bounded model-output recovery and tool-timeout settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_max_output_tokens: int = Field(default=4_096, ge=1)
    max_output_tokens: int = Field(default=16_384, ge=1)
    output_token_multiplier: float = Field(default=2.0, gt=1.0, le=4.0)
    max_output_retries: int = Field(default=1, ge=0, le=2)
    tool_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    @model_validator(mode="after")
    def maximum_must_cover_initial_limit(self) -> "ErrorRecoverySettings":
        if self.max_output_tokens < self.initial_max_output_tokens:
            raise ValueError("max output tokens must not be below the initial limit")
        return self


class LoggingSettings(BaseModel):
    """应用日志配置。

    Application logging settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.CONSOLE


class AgentLoopSettings(BaseModel):
    """Agent Loop 的有界模型迭代配置。

    Bounded model-iteration settings for the agent loop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=16, ge=1, le=100)


class RuntimePathSettings(BaseModel):
    """Agent 可读取、默认信任和写入的运行目录配置。

    Runtime directories the agent may read, trusts by default, and writes to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_root: Path = Path(".")
    knowledge_root: Path = Path("src/business/knowledge_assistant/knowledge")
    artifact_root: Path = Path("artifacts")
    user_data_root: Path = Path("users")


class ContextCompactSettings(BaseModel):
    """按模型窗口调整的 Context Compact 配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_context_characters: int = Field(default=400_000, ge=1_000)
    max_messages: int = Field(default=120, ge=8)
    keep_recent_messages: int = Field(default=20, ge=4)
    keep_recent_tool_results: int = Field(default=3, ge=1)
    max_tool_result_characters: int = Field(default=40_000, ge=1_000)
    tool_result_preview_characters: int = Field(default=2_000, ge=100)
    micro_compact_min_characters: int = Field(default=500, ge=1)
    max_summary_source_characters: int = Field(default=120_000, ge=1_000)
    max_summary_characters: int = Field(default=8_000, ge=500)
    max_reactive_retries: int = Field(default=1, ge=0, le=1)


class MemorySettings(BaseModel):
    """用户级跨会话 Memory 的容量和 Prompt 预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_memories: int = Field(default=200, ge=1, le=1_000)
    max_selected_memories: int = Field(default=5, ge=1, le=20)
    max_memory_content_characters: int = Field(default=4_096, ge=100)
    max_memory_index_characters: int = Field(default=25_000, ge=500)


class CheckpointSettings(BaseModel):
    """本地 SQLite Checkpoint 和 Conversation 索引配置。

    Local SQLite checkpoint and conversation-index settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_database_filename: str = Field(
        default="checkpoints.sqlite3",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    conversation_database_path: Path = Path(".agent/conversations.sqlite3")
    busy_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)


class TaskSystemSettings(BaseModel):
    """用户级 SQLite Task Store 配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    database_filename: str = Field(
        default="tasks.sqlite3",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    busy_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)


class RAGSettings(BaseModel):
    """本地 Embedding、切分、检索和 pgvector 配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    database_url: SecretStr | None = Field(default=None, repr=False)
    collection_name: str = Field(
        default="knowledge_assistant",
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    knowledge_base_id: str = Field(
        default="knowledge_assistant",
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)
    embedding_dimension: int = Field(default=512, gt=0, le=65_535)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_context_characters: int = Field(default=12_000, ge=100, le=1_000_000)
    chunk_size: int = Field(default=1_200, ge=100, le=100_000)
    chunk_overlap: int = Field(default=150, ge=0, le=10_000)

    @model_validator(mode="after")
    def overlap_must_be_smaller_than_chunk(self) -> "RAGSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG chunk overlap must be smaller than chunk size")
        return self


class MCPServerSettings(BaseModel):
    """一个 MCP Server 的传输和凭据配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: Literal["stdio", "http", "streamable_http", "sse", "websocket"]
    command: str | None = Field(default=None, min_length=1)
    args: tuple[str, ...] = ()
    url: str | None = Field(default=None, min_length=1)
    headers: dict[str, SecretStr] = Field(default_factory=dict, repr=False)
    env: dict[str, SecretStr] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def transport_requires_matching_endpoint(self) -> "MCPServerSettings":
        if self.transport == "stdio":
            if self.command is None:
                raise ValueError("stdio MCP server requires command")
            if self.url is not None:
                raise ValueError("stdio MCP server must not configure url")
            return self

        if self.url is None:
            raise ValueError(f"{self.transport} MCP server requires url")
        if self.command is not None or self.args or self.env:
            raise ValueError("remote MCP server must not configure command, args, or env")
        return self


class MCPSettings(BaseModel):
    """MCP 工具发现和连接超时配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    discovery_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    servers: dict[str, MCPServerSettings] = Field(default_factory=dict)


class AppSettings(BaseSettings):
    """从环境变量或本地 `.env` 文件加载的应用配置。

    Application settings loaded from environment variables or a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    model: ModelSettings = Field(default_factory=ModelSettings)
    fallback_model: ModelSettings | None = None
    model_gateway: ModelGatewaySettings = Field(default_factory=ModelGatewaySettings)
    error_recovery: ErrorRecoverySettings = Field(default_factory=ErrorRecoverySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    agent_loop: AgentLoopSettings = Field(default_factory=AgentLoopSettings)
    paths: RuntimePathSettings = Field(default_factory=RuntimePathSettings)
    context_compact: ContextCompactSettings = Field(default_factory=ContextCompactSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    task_system: TaskSystemSettings = Field(default_factory=TaskSystemSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    database_url: SecretStr | None = Field(default=None, repr=False)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """加载并缓存当前进程的应用配置。

    Load and cache application settings for the current process.
    """

    return AppSettings()


__all__ = [
    "AppSettings",
    "AgentLoopSettings",
    "CheckpointSettings",
    "ContextCompactSettings",
    "ErrorRecoverySettings",
    "Environment",
    "LogFormat",
    "LogLevel",
    "LoggingSettings",
    "MemorySettings",
    "MCPServerSettings",
    "MCPSettings",
    "ModelGatewaySettings",
    "ModelSettings",
    "RuntimePathSettings",
    "RAGSettings",
    "TaskSystemSettings",
    "get_settings",
]
