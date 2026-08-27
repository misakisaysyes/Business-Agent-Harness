"""配置加载与安全默认值测试。

Tests for configuration loading and secure defaults.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from services.config import AppSettings, Environment, LogFormat, LogLevel


def test_settings_do_not_hard_code_external_credentials_or_addresses() -> None:
    """默认配置不得包含外部凭据或连接地址。

    Default settings must not contain external credentials or connection addresses.
    """

    settings = AppSettings(_env_file=None)

    assert settings.model.provider is None
    assert settings.model.model_id is None
    assert settings.model.api_key is None
    assert settings.database_url is None


def test_settings_load_nested_values_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """嵌套配置应从统一前缀的环境变量加载。

    Nested settings should load from consistently prefixed environment variables.
    """

    monkeypatch.setenv("AGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("AGENT_MODEL__PROVIDER", "fake")
    monkeypatch.setenv("AGENT_MODEL__MODEL_ID", "fake-model")
    monkeypatch.setenv("AGENT_MODEL__API_KEY", "test-secret")
    monkeypatch.setenv("AGENT_MODEL__MAX_CONCURRENCY", "2")
    monkeypatch.setenv("AGENT_MODEL__QUOTA_SCOPE", "test-primary")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL__PROVIDER", "deepseek")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL__MODEL_ID", "fallback-model")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL__API_KEY", "fallback-secret")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL__BASE_URL", "https://fallback.test")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY__MAX_RETRIES", "2")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY__RATE_LIMIT_MAX_RETRIES", "1")
    monkeypatch.setenv("AGENT_MODEL_GATEWAY__RETRY_JITTER_SECONDS", "0.25")
    monkeypatch.setenv("AGENT_ERROR_RECOVERY__INITIAL_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("AGENT_ERROR_RECOVERY__MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("AGENT_ERROR_RECOVERY__TOOL_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("AGENT_LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("AGENT_LOGGING__FORMAT", "json")
    monkeypatch.setenv("AGENT_AGENT_LOOP__MAX_ITERATIONS", "20")
    monkeypatch.setenv("AGENT_PATHS__WORKSPACE_ROOT", "/tmp/agent-workspace")
    monkeypatch.setenv("AGENT_PATHS__KNOWLEDGE_ROOT", "knowledge")
    monkeypatch.setenv("AGENT_PATHS__ARTIFACT_ROOT", "artifacts")
    monkeypatch.setenv("AGENT_PATHS__USER_DATA_ROOT", "users")
    monkeypatch.setenv("AGENT_CONTEXT_COMPACT__MAX_CONTEXT_CHARACTERS", "240000")
    monkeypatch.setenv("AGENT_CONTEXT_COMPACT__KEEP_RECENT_MESSAGES", "12")
    monkeypatch.setenv("AGENT_MEMORY__MAX_MEMORIES", "50")
    monkeypatch.setenv("AGENT_MEMORY__MAX_SELECTED_MEMORIES", "3")
    monkeypatch.setenv("AGENT_TASK_SYSTEM__DATABASE_FILENAME", "work-items.sqlite3")
    monkeypatch.setenv("AGENT_TASK_SYSTEM__BUSY_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("AGENT_MCP__ENABLED", "true")
    monkeypatch.setenv("AGENT_MCP__DISCOVERY_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv(
        "AGENT_MCP__SERVERS",
        '{"docs":{"transport":"http","url":"https://mcp.test/api",'
        '"headers":{"Authorization":"Bearer test-mcp-secret"}}}',
    )
    monkeypatch.setenv("AGENT_DATABASE_URL", "sqlite:///test.db")

    settings = AppSettings(_env_file=None)

    assert settings.environment is Environment.TEST
    assert settings.model.provider == "fake"
    assert settings.model.model_id == "fake-model"
    assert settings.model.api_key is not None
    assert settings.model.api_key.get_secret_value() == "test-secret"
    assert settings.model.max_concurrency == 2
    assert settings.model.quota_scope == "test-primary"
    assert settings.fallback_model is not None
    assert settings.fallback_model.provider == "deepseek"
    assert settings.fallback_model.api_key is not None
    assert settings.fallback_model.api_key.get_secret_value() == "fallback-secret"
    assert settings.model_gateway.max_retries == 2
    assert settings.model_gateway.rate_limit_max_retries == 1
    assert settings.model_gateway.retry_jitter_seconds == 0.25
    assert settings.error_recovery.initial_max_output_tokens == 2_048
    assert settings.error_recovery.max_output_tokens == 8_192
    assert settings.error_recovery.tool_timeout_seconds == 15
    assert settings.logging.level is LogLevel.DEBUG
    assert settings.logging.format is LogFormat.JSON
    assert settings.agent_loop.max_iterations == 20
    assert settings.paths.workspace_root == Path("/tmp/agent-workspace")
    assert settings.paths.knowledge_root == Path("knowledge")
    assert settings.paths.artifact_root == Path("artifacts")
    assert settings.paths.user_data_root == Path("users")
    assert settings.context_compact.max_context_characters == 240_000
    assert settings.context_compact.keep_recent_messages == 12
    assert settings.memory.max_memories == 50
    assert settings.memory.max_selected_memories == 3
    assert settings.task_system.database_filename == "work-items.sqlite3"
    assert settings.task_system.busy_timeout_seconds == 8
    assert settings.mcp.enabled
    assert settings.mcp.discovery_timeout_seconds == 12
    assert settings.mcp.servers["docs"].url == "https://mcp.test/api"
    assert (
        settings.mcp.servers["docs"].headers["Authorization"].get_secret_value()
        == "Bearer test-mcp-secret"
    )
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == "sqlite:///test.db"


def test_secret_values_are_masked_in_settings_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置的字符串表示不得泄露密钥或数据库地址。

    Settings representations must not leak API keys or database addresses.
    """

    monkeypatch.setenv("AGENT_MODEL__API_KEY", "sensitive-api-key")
    monkeypatch.setenv("AGENT_DATABASE_URL", "postgresql://user:password@localhost/db")
    monkeypatch.setenv(
        "AGENT_MCP__SERVERS",
        '{"docs":{"transport":"http","url":"https://mcp.test/api",'
        '"headers":{"Authorization":"sensitive-mcp-token"}}}',
    )

    settings = AppSettings(_env_file=None)
    settings_repr = repr(settings)

    assert "sensitive-api-key" not in settings_repr
    assert "postgresql://user:password@localhost/db" not in settings_repr
    assert "sensitive-mcp-token" not in settings_repr


def test_invalid_environment_configuration_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不支持的环境配置应产生明确的校验错误。

    Unsupported environment configuration should produce an explicit validation error.
    """

    monkeypatch.setenv("AGENT_LOGGING__LEVEL", "VERBOSE")

    with pytest.raises(ValidationError, match="AGENT_LOGGING__LEVEL|logging.level"):
        AppSettings(_env_file=None)
