"""多用户 HTTP CLI 测试。

Multi-user HTTP CLI tests.
"""

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import entrypoints.cli as cli_module
from entrypoints.cli import (
    AgentServerClient,
    ConversationDetail,
    ConversationInfo,
    ConversationSummary,
    MCPServerSummary,
    MCPStatus,
    MCPToolSummary,
    SkillSummary,
    app,
    show_model_gateway_event,
)
from services.config import LogFormat, LoggingSettings, LogLevel
from services.model_gateway import ModelGatewayEvent, ModelGatewayEventType


class FakeAgentServerClient:
    """不访问网络的 CLI HTTP Client 替身。"""

    created_with: tuple[str, str] | None = None
    closed = False
    sent_messages: list[tuple[str, str, str | None]] = []

    def __init__(self, server_url: str, user_id: str) -> None:
        type(self).created_with = (server_url, user_id)

    def close(self) -> None:
        type(self).closed = True

    def create_conversation(self) -> ConversationInfo:
        return ConversationInfo(
            conversation_id="conversation-001",
            agent_name="knowledge_assistant",
            primary_model="moonshot/kimi-k3",
        )

    def send_message(
        self,
        conversation_id: str,
        content: str,
        required_tool: str | None = None,
    ) -> dict[str, Any]:
        type(self).sent_messages.append((conversation_id, content, required_tool))
        return {
            "conversation_id": conversation_id,
            "run_id": "run-001",
            "status": "idle",
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "固定回答"},
            ],
            "permission_request": None,
            "model_events": [
                {
                    "event_type": "selected",
                    "model": "moonshot/kimi-k3",
                    "reason": "model request succeeded",
                    "retry_number": None,
                    "max_retries": None,
                    "delay_seconds": None,
                    "fallback_model": None,
                }
            ],
        }

    def list_conversations(self) -> tuple[ConversationSummary, ...]:
        return (
            ConversationSummary(
                conversation_id="conversation-001",
                status="idle",
            ),
        )

    def delete_conversation(self, conversation_id: str) -> None:
        raise AssertionError("conversation deletion should not be called")

    def get_token_usage(self) -> dict[str, Any]:
        return {
            "user_id": "alice",
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        }

    def list_skills(self) -> tuple[SkillSummary, ...]:
        return (
            SkillSummary("knowledge-synthesis", "Synthesize knowledge."),
            SkillSummary("alice-workflow", "Alice private workflow."),
        )

    def get_mcp_status(self) -> MCPStatus:
        return MCPStatus(
            enabled=True,
            servers=(
                MCPServerSummary(
                    name="demo.server",
                    status="connected",
                    tools=("mcp__demo_server__lookup", "mcp__demo_server__erase"),
                    error_type=None,
                ),
            ),
            tools=(
                MCPToolSummary(
                    name="mcp__demo_server__lookup",
                    server_name="demo.server",
                    remote_name="lookup",
                    read_only=True,
                    destructive=False,
                ),
                MCPToolSummary(
                    name="mcp__demo_server__erase",
                    server_name="demo.server",
                    remote_name="erase",
                    read_only=False,
                    destructive=True,
                ),
            ),
        )

    def resume_permission(
        self,
        conversation_id: str,
        run_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        raise AssertionError("permission resume should not be called")


class ConversationManagementClient:
    """记录 CLI 新建、切换和删除操作的 Server Client 替身。"""

    conversations: dict[str, str] = {}
    created_count = 0
    deleted: list[str] = []

    def __init__(self, server_url: str, user_id: str) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.conversations = {}
        cls.created_count = 0
        cls.deleted = []

    def close(self) -> None:
        pass

    def create_conversation(self) -> ConversationInfo:
        type(self).created_count += 1
        conversation_id = f"conversation-{type(self).created_count:03d}"
        type(self).conversations[conversation_id] = "idle"
        return ConversationInfo(
            conversation_id=conversation_id,
            agent_name="knowledge_assistant",
            primary_model="moonshot/kimi-k3",
        )

    def list_conversations(self) -> tuple[ConversationSummary, ...]:
        return tuple(
            ConversationSummary(conversation_id=conversation_id, status=status)
            for conversation_id, status in type(self).conversations.items()
        )

    def delete_conversation(self, conversation_id: str) -> None:
        del type(self).conversations[conversation_id]
        type(self).deleted.append(conversation_id)

    def get_token_usage(self) -> dict[str, Any]:
        return {
            "user_id": "alice",
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        }

    def send_message(
        self,
        conversation_id: str,
        content: str,
        required_tool: str | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("messages should not be sent in this test")

    def resume_permission(
        self,
        conversation_id: str,
        run_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        raise AssertionError("permission resume should not be called")


class RecordingHttpClient:
    """记录 HTTPX Client 构造参数，验证本地请求不使用系统代理。"""

    options: dict[str, Any] = {}

    def __init__(self, **options: Any) -> None:
        type(self).options = options

    def close(self) -> None:
        pass


class WaitingRecoveryClient(FakeAgentServerClient):
    """模拟 Server 重启后仍在等待 Permission 的旧 Conversation。"""

    resumed: tuple[str, str, bool] | None = None

    def list_conversations(self) -> tuple[ConversationSummary, ...]:
        return (
            ConversationSummary("conversation-001", "idle"),
            ConversationSummary("persisted-waiting", "waiting_permission"),
        )

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        return ConversationDetail(
            conversation_id=conversation_id,
            status="waiting_permission",
            active_run_id="persisted-run",
            permission_request={
                "kind": "tool_permission",
                "requests": [
                    {
                        "tool_use_id": "write-1",
                        "tool_name": "report_writer",
                        "input": {"path": "report.md", "overwrite": True},
                        "reason": "overwrite requires approval",
                    }
                ],
            },
        )

    def resume_permission(
        self,
        conversation_id: str,
        run_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        type(self).resumed = (conversation_id, run_id, approved)
        return {
            "conversation_id": conversation_id,
            "run_id": run_id,
            "status": "idle",
            "messages": [{"role": "assistant", "content": "恢复完成"}],
            "permission_request": None,
            "model_events": [],
        }


class RunningCancellationClient(FakeAgentServerClient):
    """模拟另一个 CLI 中正在执行的 Conversation。"""

    cancelled: tuple[str, str] | None = None

    def list_conversations(self) -> tuple[ConversationSummary, ...]:
        return (
            ConversationSummary("conversation-001", "idle"),
            ConversationSummary("persisted-running", "running"),
        )

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        return ConversationDetail(
            conversation_id=conversation_id,
            status="running",
            active_run_id="active-run",
            permission_request=None,
        )

    def cancel_run(self, conversation_id: str, run_id: str) -> ConversationDetail:
        type(self).cancelled = (conversation_id, run_id)
        return ConversationDetail(conversation_id, "idle", None, None)


def test_agent_server_client_ignores_environment_proxy(monkeypatch) -> None:
    """本地 Agent Server 请求不得被 HTTP_PROXY 等环境变量截获。"""

    monkeypatch.setattr(cli_module.httpx, "Client", RecordingHttpClient)

    client = AgentServerClient("http://127.0.0.1:8000", "alice")
    client.close()

    assert RecordingHttpClient.options["trust_env"] is False
    assert RecordingHttpClient.options["base_url"] == "http://127.0.0.1:8000"


def test_cli_enables_terminal_line_editing(monkeypatch) -> None:
    """Chat 启动时应加载 Readline，使中文字符可以一次完整删除。"""

    imported: list[str] = []
    monkeypatch.setattr(cli_module, "import_module", imported.append)

    cli_module._enable_terminal_line_editing()

    assert imported == ["readline"]


def test_cli_allows_platforms_without_readline(monkeypatch) -> None:
    """没有 Readline 的平台应安全退回 Python 默认输入。"""

    def unavailable(module_name: str) -> None:
        raise ImportError(module_name)

    monkeypatch.setattr(cli_module, "import_module", unavailable)

    cli_module._enable_terminal_line_editing()


def test_user_prompt_is_one_protected_readline_prompt(monkeypatch) -> None:
    """完整 ``You [...]`` 必须作为不可编辑 Prompt 一次性交给 input。"""

    received_prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        received_prompts.append(prompt)
        return "hello"

    monkeypatch.setattr("builtins.input", fake_input)

    value = cli_module._read_user_message("You [abc123]")

    assert value == "hello"
    assert received_prompts == ["You [abc123]: "]


def test_serve_initializes_logging_before_starting_uvicorn(monkeypatch) -> None:
    """Server 启动时应应用日志配置，并禁止 Uvicorn 覆盖该配置。"""

    logging_settings = LoggingSettings(level=LogLevel.DEBUG, format=LogFormat.JSON)
    configured: list[LoggingSettings] = []
    uvicorn_calls: list[dict[str, Any]] = []

    class FakeSettings:
        logging = logging_settings

    monkeypatch.setattr(cli_module, "get_settings", FakeSettings)
    monkeypatch.setattr(cli_module, "configure_logging", configured.append)
    monkeypatch.setattr(
        "uvicorn.run",
        lambda application, **kwargs: uvicorn_calls.append({"application": application, **kwargs}),
    )

    cli_module.serve(host="0.0.0.0", port=9000)

    assert configured == [logging_settings]
    assert uvicorn_calls == [
        {
            "application": "entrypoints.api:app",
            "host": "0.0.0.0",
            "port": 9000,
            "log_config": None,
            "log_level": "debug",
        }
    ]


def test_cli_sends_message_through_agent_server(monkeypatch) -> None:
    """CLI 应携带用户标识，通过 Agent Server 完成单次消息。"""

    FakeAgentServerClient.created_with = None
    FakeAgentServerClient.closed = False
    monkeypatch.setattr(cli_module, "AgentServerClient", FakeAgentServerClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice", "--message", "你好"],
    )

    assert result.exit_code == 0
    assert "knowledge_assistant [on-001] [moonshot/kimi-k3]: 固定回答" in result.stdout
    assert "Fallback model:" not in result.stdout
    assert FakeAgentServerClient.created_with == (
        "http://127.0.0.1:8000",
        "alice",
    )
    assert FakeAgentServerClient.closed


def test_interactive_chat_prompt_contains_conversation_suffix(monkeypatch) -> None:
    """交互输入提示应展示 Conversation ID 后六位。"""

    monkeypatch.setattr(cli_module, "AgentServerClient", FakeAgentServerClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/quit\n",
    )

    assert result.exit_code == 0
    assert "You [on-001]:" in result.stdout


def test_cli_can_create_list_switch_and_delete_conversations(monkeypatch) -> None:
    """CLI 应支持创建、列出、按后缀切换和删除 Conversation。"""

    ConversationManagementClient.reset()
    monkeypatch.setattr(
        cli_module,
        "AgentServerClient",
        ConversationManagementClient,
    )

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/new\n/list\n/switch on-001\n/delete on-002\ny\n/quit\n",
    )

    assert result.exit_code == 0
    assert "[on-001] idle conversation-001" in result.stdout
    assert "[on-002] idle conversation-002" in result.stdout
    assert "Switched to conversation [on-001]" in result.stdout
    assert "Deleted conversation [on-002]" in result.stdout
    assert ConversationManagementClient.deleted == ["conversation-002"]


def test_cli_can_resume_persisted_permission_when_switching(monkeypatch) -> None:
    """切换到重启前暂停的 Conversation 时应继续原 Run 的审批。"""

    WaitingRecoveryClient.resumed = None
    monkeypatch.setattr(cli_module, "AgentServerClient", WaitingRecoveryClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/switch waiting\ny\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Switched to conversation [aiting]" in result.stdout
    assert "Tool: report_writer" in result.stdout
    assert "knowledge_assistant [aiting] [moonshot/kimi-k3]: 恢复完成" in result.stdout
    assert WaitingRecoveryClient.resumed == (
        "persisted-waiting",
        "persisted-run",
        True,
    )


def test_cli_can_cancel_a_running_conversation_from_another_terminal(monkeypatch) -> None:
    """第二个 CLI 应能按 Conversation 后缀取消活跃 Run。"""

    RunningCancellationClient.cancelled = None
    monkeypatch.setattr(cli_module, "AgentServerClient", RunningCancellationClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/cancel running\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Cancelled run in conversation [unning]" in result.stdout
    assert RunningCancellationClient.cancelled == ("persisted-running", "active-run")


def test_cli_shows_current_user_token_usage(monkeypatch) -> None:
    """`/usage` 应只展示当前 CLI 用户的累计 Token 用量。"""

    ConversationManagementClient.reset()
    monkeypatch.setattr(cli_module, "AgentServerClient", ConversationManagementClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/usage\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Token usage: input=10, output=2, total=12" in result.stdout


def test_cli_lists_skills_and_mcp_tools(monkeypatch) -> None:
    """`/skills` 和 `/mcp` 应展示服务端实际返回的能力清单。"""

    monkeypatch.setattr(cli_module, "AgentServerClient", FakeAgentServerClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/skills\n/mcp\n/quit\n",
    )

    assert result.exit_code == 0
    assert "knowledge-synthesis: Synthesize knowledge." in result.stdout
    assert "alice-workflow: Alice private workflow." in result.stdout
    assert "demo.server [connected]" in result.stdout
    assert "mcp__demo_server__lookup [readOnly]" in result.stdout
    assert "mcp__demo_server__erase [destructive]" in result.stdout


def test_cli_test_command_forces_tool_skill_and_mcp(monkeypatch) -> None:
    """`/test` 应按类别设置 required_tool，并在 Skill 测试中固定名称。"""

    FakeAgentServerClient.sent_messages = []
    monkeypatch.setattr(cli_module, "AgentServerClient", FakeAgentServerClient)

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input=(
            "/test tool calculator expression 设置为 1+1\n"
            "/test skill knowledge-synthesis 总结 sample.txt\n"
            "/test mcp mcp__demo_server__lookup query 设置为 MCP-01\n"
            "/quit\n"
        ),
    )

    assert result.exit_code == 0
    assert [item[2] for item in FakeAgentServerClient.sent_messages] == [
        "calculator",
        "load_skill",
        "mcp__demo_server__lookup",
    ]
    assert 'name 严格设置为 "knowledge-synthesis"' in FakeAgentServerClient.sent_messages[1][1]


def test_cli_creates_a_replacement_after_deleting_current_conversation(
    monkeypatch,
) -> None:
    """删除当前 Conversation 后应自动创建新会话，保证 CLI 仍可继续使用。"""

    ConversationManagementClient.reset()
    monkeypatch.setattr(
        cli_module,
        "AgentServerClient",
        ConversationManagementClient,
    )

    result = CliRunner().invoke(
        app,
        ["chat", "--user", "alice"],
        input="/delete on-001\ny\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Deleted conversation [on-001]" in result.stdout
    assert "Conversation: conversation-002" in result.stdout
    assert "You [on-002]:" in result.stdout
    assert ConversationManagementClient.deleted == ["conversation-001"]


def test_cli_shows_model_retry_and_fallback_events(capsys) -> None:
    """CLI 应向用户展示重试次数和模型降级信息。"""

    show_model_gateway_event(
        ModelGatewayEvent(
            event_type=ModelGatewayEventType.RETRY,
            model="moonshot/kimi-k3",
            reason="TimeoutError",
            retry_number=2,
            max_retries=2,
            delay_seconds=2.5,
        )
    )
    show_model_gateway_event(
        ModelGatewayEvent(
            event_type=ModelGatewayEventType.FALLBACK,
            model="moonshot/kimi-k3",
            fallback_model="deepseek/deepseek-v4-flash",
            reason="primary reached its concurrency limit",
        )
    )

    output = capsys.readouterr().err

    assert "[模型重试 2/2]" in output
    assert "2.50s" in output
    assert "[模型降级]" in output
    assert "deepseek/deepseek-v4-flash" in output


def test_cli_does_not_print_selected_event_as_a_separate_line(capsys) -> None:
    """成功路由事件只用于回答前缀，不单独打印。"""

    show_model_gateway_event(
        ModelGatewayEvent(
            event_type=ModelGatewayEventType.SELECTED,
            model="moonshot/kimi-k3",
            reason="model request succeeded",
        )
    )

    assert capsys.readouterr().err == ""


def test_cli_shows_gateway_events_before_final_server_failure(capsys) -> None:
    """503 响应应先展示重试过程，再展示最终失败信息。"""

    response = httpx.Response(
        503,
        json={
            "detail": {
                "message": "all model routes failed",
                "model_events": [
                    {
                        "event_type": "retry",
                        "model": "moonshot/kimi-k3",
                        "reason": "TimeoutError",
                        "retry_number": 2,
                        "max_retries": 2,
                        "delay_seconds": 2.0,
                        "fallback_model": None,
                    }
                ],
            }
        },
    )

    with pytest.raises(RuntimeError, match="all model routes failed"):
        AgentServerClient._raise_for_status(response)

    assert "[模型重试 2/2]" in capsys.readouterr().err
