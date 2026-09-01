"""多用户 Agent Server 命令行客户端。

Multi-user command-line client for the agent server.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer

from business.knowledge_assistant.search_routing import SearchMode
from services.config import get_settings
from services.model_gateway import ModelGatewayEvent, ModelGatewayEventType
from services.observability import configure_logging

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

app = typer.Typer(help="Run the configured business agent.")


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    """Agent Server 创建的 Conversation 及运行身份信息。

    Conversation and runtime identity created by the agent server.
    """

    conversation_id: str
    agent_name: str
    primary_model: str


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """CLI 用于列出和切换的 Conversation 摘要。

    Conversation summary used by the CLI for listing and switching.
    """

    conversation_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ConversationDetail:
    """恢复持久化 Permission interrupt 所需的会话详情。"""

    conversation_id: str
    status: str
    active_run_id: str | None
    permission_request: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class SkillSummary:
    """CLI 展示的 Skill 名称和描述。"""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class MCPToolSummary:
    """CLI 展示和测试的 MCP Tool 元数据。"""

    name: str
    server_name: str
    remote_name: str
    read_only: bool
    destructive: bool


@dataclass(frozen=True, slots=True)
class MCPServerSummary:
    """CLI 展示的 MCP Server 发现状态。"""

    name: str
    status: str
    tools: tuple[str, ...]
    error_type: str | None


@dataclass(frozen=True, slots=True)
class MCPStatus:
    """Agent Server 返回的 MCP 总体状态。"""

    enabled: bool
    servers: tuple[MCPServerSummary, ...]
    tools: tuple[MCPToolSummary, ...]


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """CLI 展示的模型配置。"""

    name: str
    primary: bool


@dataclass(frozen=True, slots=True)
class TestCommand:
    """解析后的 `/test` 类型、目标和测试请求。"""

    kind: str
    target: str
    prompt: str


def show_model_gateway_event(event: ModelGatewayEvent) -> None:
    """在 Chat 中展示模型重试和降级信息。"""

    _show_model_gateway_event(
        event_type=event.event_type.value,
        model=event.model,
        reason=event.reason,
        retry_number=event.retry_number,
        max_retries=event.max_retries,
        delay_seconds=event.delay_seconds,
        fallback_model=event.fallback_model,
    )


def _show_model_gateway_event(
    event_type: str,
    model: str,
    reason: str,
    retry_number: int | None = None,
    max_retries: int | None = None,
    delay_seconds: float | None = None,
    fallback_model: str | None = None,
) -> None:
    if event_type == ModelGatewayEventType.SELECTED.value:
        return

    if event_type == ModelGatewayEventType.RETRY.value:
        typer.echo(
            f"[模型重试 {retry_number or 0}/{max_retries or 0}] "
            f"{model} 将在 {delay_seconds or 0.0:.2f}s 后重试：{reason}",
            err=True,
        )
        return

    typer.echo(f"[模型降级] {model} → {fallback_model}：{reason}", err=True)


class AgentServerClient:
    """封装 CLI 使用的最小 Agent Server HTTP 调用。"""

    def __init__(self, server_url: str, user_id: str) -> None:
        self.user_id = user_id
        self._client = httpx.Client(
            base_url=server_url.rstrip("/"),
            timeout=120.0,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def create_conversation(self) -> ConversationInfo:
        response = self._client.post(f"/users/{self.user_id}/conversations")
        self._raise_for_status(response)
        body = response.json()
        return ConversationInfo(
            conversation_id=str(body["conversation_id"]),
            agent_name=str(body["agent_name"]),
            primary_model=str(body["primary_model"]),
        )

    def send_message(
        self,
        conversation_id: str,
        content: str,
        required_tool: str | None = None,
        search_mode: SearchMode = SearchMode.AUTO,
        model: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, str] = {"content": content, "search_mode": search_mode.value}
        if required_tool is not None:
            body["required_tool"] = required_tool
        if model is not None:
            body["model"] = model
        response = self._client.post(
            f"/users/{self.user_id}/conversations/{conversation_id}/messages",
            json=body,
        )
        self._raise_for_status(response)
        return response.json()

    def list_conversations(self) -> tuple[ConversationSummary, ...]:
        response = self._client.get(f"/users/{self.user_id}/conversations")
        self._raise_for_status(response)
        return tuple(
            ConversationSummary(
                conversation_id=str(item["conversation_id"]),
                status=str(item["status"]),
            )
            for item in response.json().get("conversations", [])
        )

    def delete_conversation(self, conversation_id: str) -> None:
        response = self._client.delete(f"/users/{self.user_id}/conversations/{conversation_id}")
        self._raise_for_status(response)

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        response = self._client.get(f"/users/{self.user_id}/conversations/{conversation_id}")
        self._raise_for_status(response)
        body = response.json()
        return ConversationDetail(
            conversation_id=str(body["conversation_id"]),
            status=str(body["status"]),
            active_run_id=(
                str(body["active_run_id"]) if body.get("active_run_id") is not None else None
            ),
            permission_request=cast(dict[str, Any] | None, body.get("permission_request")),
        )

    def get_token_usage(self) -> dict[str, Any]:
        """读取当前可信本地用户的累计模型 Token 用量。"""

        response = self._client.get(f"/users/{self.user_id}/usage")
        self._raise_for_status(response)
        return cast(dict[str, Any], response.json())

    def get_models(self) -> tuple[ModelSummary, ...]:
        """读取服务端当前允许选择的模型。"""

        response = self._client.get(f"/users/{self.user_id}/models")
        self._raise_for_status(response)
        return tuple(
            ModelSummary(name=str(item["name"]), primary=bool(item["primary"]))
            for item in response.json().get("models", [])
        )

    def list_skills(self) -> tuple[SkillSummary, ...]:
        """读取当前用户可见的内置和私有 Skill。"""

        response = self._client.get(f"/users/{self.user_id}/skills")
        self._raise_for_status(response)
        return tuple(
            SkillSummary(
                name=str(item["name"]),
                description=str(item["description"]),
            )
            for item in response.json().get("skills", [])
        )

    def get_mcp_status(self) -> MCPStatus:
        """读取 Agent Server 实际发现的 MCP Server 和 Tool。"""

        response = self._client.get(f"/users/{self.user_id}/mcp")
        self._raise_for_status(response)
        body = response.json()
        return MCPStatus(
            enabled=bool(body["enabled"]),
            servers=tuple(
                MCPServerSummary(
                    name=str(item["name"]),
                    status=str(item["status"]),
                    tools=tuple(str(name) for name in item.get("tools", [])),
                    error_type=(
                        str(item["error_type"]) if item.get("error_type") is not None else None
                    ),
                )
                for item in body.get("servers", [])
            ),
            tools=tuple(
                MCPToolSummary(
                    name=str(item["name"]),
                    server_name=str(item["server_name"]),
                    remote_name=str(item["remote_name"]),
                    read_only=bool(item["read_only"]),
                    destructive=bool(item["destructive"]),
                )
                for item in body.get("tools", [])
            ),
        )

    def resume_permission(
        self,
        conversation_id: str,
        run_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/users/{self.user_id}/conversations/{conversation_id}/runs/{run_id}/permission",
            json={"approved": approved},
        )
        self._raise_for_status(response)
        return response.json()

    def cancel_run(
        self,
        conversation_id: str,
        run_id: str,
    ) -> ConversationDetail:
        """请求 Server 取消正在执行的 Run。"""

        response = self._client.post(
            f"/users/{self.user_id}/conversations/{conversation_id}/runs/{run_id}/cancel"
        )
        self._raise_for_status(response)
        body = response.json()
        return ConversationDetail(
            conversation_id=str(body["conversation_id"]),
            status=str(body["status"]),
            active_run_id=None,
            permission_request=None,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if not response.is_error:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        if isinstance(detail, dict):
            detail_data = cast(dict[str, Any], detail)
            _show_model_events(detail_data)
            message = detail_data.get("message", detail_data)
        else:
            message = detail
        raise RuntimeError(f"Agent Server returned HTTP {response.status_code}: {message}")


@app.callback()
def root() -> None:
    """提供 Agent Server 和 Chat Client 命令。"""


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """启动单进程 Agent Server。"""

    import uvicorn

    settings = get_settings()
    configure_logging(settings.logging)
    uvicorn.run(
        "entrypoints.api:app",
        host=host,
        port=port,
        log_config=None,
        log_level=settings.logging.level.value.lower(),
        access_log=False,
    )


@app.command("index")
def index_command(
    source: Annotated[Path, typer.Option("--source", exists=True, file_okay=False)],
    scope: Annotated[Literal["public", "user"], typer.Option("--scope")] = "public",
    user: Annotated[str | None, typer.Option("--user")] = None,
    rebuild: Annotated[bool, typer.Option("--rebuild")] = False,
) -> None:
    """把 Markdown/TXT 资料增量写入配置的 pgvector Collection。"""

    from entrypoints.indexer import index_documents

    settings = get_settings()
    configure_logging(settings.logging)
    if scope == "user" and not user:
        raise typer.BadParameter("--user is required when --scope=user")
    if scope == "public" and user is not None:
        raise typer.BadParameter("--user is only valid when --scope=user")
    report = index_documents(
        settings,
        source,
        scope=scope,
        user_id=user,
        rebuild=rebuild,
    )
    typer.echo(
        f"indexed={report.indexed} skipped={report.skipped} "
        f"deleted_chunks={report.deleted_chunks} failed={len(report.failed)}"
    )
    for failure in report.failed:
        typer.echo(f"failed source={failure.source}: {failure.error}", err=True)
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def chat(
    user: str = typer.Option(..., "--user", help="Trusted local user ID."),
    message: str | None = None,
    server_url: str = typer.Option(DEFAULT_SERVER_URL, "--server-url"),
) -> None:
    """进入 Conversation Loop；只有 `/exit` 或 `/quit` 才退出。"""

    _enable_terminal_line_editing()
    client = AgentServerClient(server_url, user)
    try:
        conversation = client.create_conversation()
        conversation_id = conversation.conversation_id
        search_mode = SearchMode.AUTO
        selected_model: str | None = None
        typer.echo(f"Conversation: {conversation_id}")
        if message is not None:
            _execute_message(
                client,
                conversation,
                message,
                search_mode=search_mode,
                model=selected_model,
            )
            return

        typer.echo(
            "输入 /new 新建、/list 查看、/switch <ID> 切换、"
            "/delete <ID> 删除、/cancel <ID> 取消运行、/usage 查看用量、"
            "/model [模型名] 查看或切换主模型、"
            "/search-mode auto|rag|web|hybrid 设置检索模式、"
            "/skills 查看 Skill、/mcp 查看 MCP、/test 测试能力、"
            "/exit 或 /quit 退出。"
        )
        while True:
            user_message = _read_user_message(f"You [{_conversation_suffix(conversation_id)}]")
            command = user_message.strip().lower()
            if command in {"/exit", "/quit"}:
                return
            if command == "/new":
                conversation = client.create_conversation()
                conversation_id = conversation.conversation_id
                search_mode = SearchMode.AUTO
                typer.echo(f"Conversation: {conversation_id}")
                continue
            if command.startswith("/search-mode"):
                selected_mode = _parse_search_mode(user_message)
                if selected_mode is None:
                    typer.echo("Usage: /search-mode auto|rag|web|hybrid", err=True)
                else:
                    search_mode = selected_mode
                    typer.echo(f"Search mode: {search_mode.value}")
                continue
            if command == "/model" or command.startswith("/model "):
                model_name = _command_argument(user_message, "/model")
                models = client.get_models()
                if model_name is None:
                    _show_models(models, selected_model or conversation.primary_model)
                    continue
                configured_models = {model.name for model in models}
                if model_name not in configured_models:
                    typer.echo(
                        f"Error: model is not configured: {model_name}",
                        err=True,
                    )
                    _show_models(models, selected_model or conversation.primary_model)
                    continue
                selected_model = model_name
                typer.echo(f"Model: {selected_model}")
                continue
            if command == "/list":
                _show_conversations(client.list_conversations(), conversation_id)
                continue
            if command == "/usage":
                usage = client.get_token_usage().get("token_usage", {})
                typer.echo(
                    "Token usage: "
                    f"input={usage.get('input_tokens', 0)}, "
                    f"output={usage.get('output_tokens', 0)}, "
                    f"total={usage.get('total_tokens', 0)}"
                )
                continue
            if command == "/skills":
                _show_skills(client.list_skills())
                continue
            if command == "/mcp":
                _show_mcp(client.get_mcp_status())
                continue
            if command.startswith("/test"):
                test = _parse_test_command(user_message)
                if test is None:
                    typer.echo(
                        "Usage: /test tool|skill|mcp <name> <test request>",
                        err=True,
                    )
                    continue
                required_tool = test.target
                if test.kind == "skill":
                    installed_skills = {skill.name for skill in client.list_skills()}
                    if test.target not in installed_skills:
                        typer.echo(f"Error: skill is not installed: {test.target}", err=True)
                        continue
                    required_tool = "load_skill"
                elif test.kind == "mcp":
                    installed_mcp_tools = {tool.name for tool in client.get_mcp_status().tools}
                    if test.target not in installed_mcp_tools:
                        typer.echo(
                            f"Error: MCP tool is not installed: {test.target}",
                            err=True,
                        )
                        continue
                content = _build_test_prompt(test)
                _execute_message(
                    client,
                    conversation,
                    content,
                    required_tool=required_tool,
                    search_mode=search_mode,
                    model=selected_model,
                )
                continue
            if command.startswith("/switch"):
                reference = _command_argument(user_message, "/switch")
                if reference is None:
                    typer.echo("Usage: /switch <conversation ID or last 6 characters>")
                    continue
                try:
                    target = _resolve_conversation(
                        reference,
                        client.list_conversations(),
                    )
                except ValueError as error:
                    typer.echo(f"Error: {error}", err=True)
                    continue
                if target.status == "running":
                    typer.echo(
                        f"Error: conversation [{_conversation_suffix(target.conversation_id)}] "
                        f"is {target.status}",
                        err=True,
                    )
                    continue
                conversation = ConversationInfo(
                    conversation_id=target.conversation_id,
                    agent_name=conversation.agent_name,
                    primary_model=conversation.primary_model,
                )
                conversation_id = target.conversation_id
                search_mode = SearchMode.AUTO
                typer.echo(f"Switched to conversation [{_conversation_suffix(conversation_id)}]")
                if target.status == "waiting_permission":
                    detail = client.get_conversation(conversation_id)
                    if detail.active_run_id is None or detail.permission_request is None:
                        typer.echo("Error: persisted permission state is incomplete", err=True)
                        continue
                    _complete_run(
                        client,
                        conversation,
                        {
                            "conversation_id": conversation_id,
                            "run_id": detail.active_run_id,
                            "status": detail.status,
                            "messages": [],
                            "permission_request": detail.permission_request,
                            "model_events": [],
                        },
                    )
                continue
            if command.startswith("/delete"):
                reference = _command_argument(user_message, "/delete")
                if reference is None:
                    typer.echo("Usage: /delete <conversation ID or last 6 characters>")
                    continue
                try:
                    target = _resolve_conversation(
                        reference,
                        client.list_conversations(),
                    )
                except ValueError as error:
                    typer.echo(f"Error: {error}", err=True)
                    continue
                suffix = _conversation_suffix(target.conversation_id)
                if not typer.confirm(f"Delete conversation [{suffix}]?", default=False):
                    continue
                try:
                    client.delete_conversation(target.conversation_id)
                except RuntimeError as error:
                    typer.echo(f"Error: {error}", err=True)
                    continue
                typer.echo(f"Deleted conversation [{suffix}]")
                if target.conversation_id == conversation_id:
                    conversation = client.create_conversation()
                    conversation_id = conversation.conversation_id
                    search_mode = SearchMode.AUTO
                    typer.echo(f"Conversation: {conversation_id}")
                continue
            if command.startswith("/cancel"):
                reference = _command_argument(user_message, "/cancel")
                if reference is None:
                    typer.echo("Usage: /cancel <conversation ID or last 6 characters>")
                    continue
                try:
                    target = _resolve_conversation(
                        reference,
                        client.list_conversations(),
                    )
                    detail = client.get_conversation(target.conversation_id)
                    if detail.status != "running" or detail.active_run_id is None:
                        raise ValueError("conversation does not have a running run")
                    client.cancel_run(target.conversation_id, detail.active_run_id)
                except (RuntimeError, ValueError) as error:
                    typer.echo(f"Error: {error}", err=True)
                    continue
                typer.echo(
                    f"Cancelled run in conversation "
                    f"[{_conversation_suffix(target.conversation_id)}]"
                )
                continue
            _execute_message(
                client,
                conversation,
                user_message,
                search_mode=search_mode,
                model=selected_model,
            )
    except (httpx.HTTPError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        client.close()


def _enable_terminal_line_editing() -> None:
    """加载 Readline/Libedit，使终端按完整 Unicode 字符编辑输入。

    Load Readline/Libedit so terminal input is edited by complete Unicode characters.
    """

    try:
        import_module("readline")
    except ImportError:
        # Windows 等没有 readline 的环境继续使用 Python 默认输入。
        # Environments without readline, such as Windows, keep the default input.
        return


def _read_user_message(prompt: str) -> str:
    """把完整提示符交给 Readline，防止退格擦除 ``You [...]``。

    Pass the complete prompt to Readline so backspace cannot erase ``You [...]``.

    Click/Typer 会先绘制提示符主体，再只把最后一个字符传给 ``input()``。
    在 macOS Libedit 中，这会让退格键在视觉上越过输入边界。直接把完整提示符
    作为 ``input()`` 的 prompt 后，Readline 会把它视为不可编辑区域。
    """

    try:
        return input(f"{prompt}: ")
    except (EOFError, KeyboardInterrupt) as error:
        typer.echo()
        raise typer.Abort() from error


def _execute_message(
    client: AgentServerClient,
    conversation: ConversationInfo,
    content: str,
    required_tool: str | None = None,
    search_mode: SearchMode = SearchMode.AUTO,
    model: str | None = None,
) -> None:
    conversation_id = conversation.conversation_id
    if model is not None:
        result = client.send_message(
            conversation_id,
            content,
            required_tool=required_tool,
            search_mode=search_mode,
            model=model,
        )
    elif required_tool is None and search_mode is SearchMode.AUTO:
        result = client.send_message(conversation_id, content)
    elif required_tool is None:
        result = client.send_message(conversation_id, content, search_mode=search_mode)
    elif search_mode is SearchMode.AUTO:
        result = client.send_message(
            conversation_id,
            content,
            required_tool=required_tool,
        )
    else:
        result = client.send_message(
            conversation_id,
            content,
            required_tool=required_tool,
            search_mode=search_mode,
        )
    _complete_run(client, conversation, result)


def _show_skills(skills: tuple[SkillSummary, ...]) -> None:
    """展示当前用户可加载的 Skill。"""

    if not skills:
        typer.echo("No skills installed.")
        return
    typer.echo("Installed skills:")
    for skill in skills:
        typer.echo(f"- {skill.name}: {skill.description}")


def _show_models(models: tuple[ModelSummary, ...], selected_model: str) -> None:
    """展示模型清单和当前 CLI 会话选择。"""

    if not models:
        typer.echo("No models configured.")
        return
    typer.echo(f"Current model: {selected_model}")
    typer.echo("Available models:")
    for model in models:
        suffix = " (primary)" if model.primary else ""
        marker = " *" if model.name == selected_model else ""
        typer.echo(f"- {model.name}{suffix}{marker}")


def _show_mcp(status: MCPStatus) -> None:
    """展示 MCP 开关、Server 状态和 Tool 安全属性。"""

    if not status.enabled:
        typer.echo("MCP is disabled.")
        return
    if not status.servers:
        typer.echo("No MCP servers configured.")
        return
    typer.echo("MCP servers:")
    tools = {tool.name: tool for tool in status.tools}
    for server in status.servers:
        suffix = f" ({server.error_type})" if server.error_type else ""
        typer.echo(f"- {server.name} [{server.status}]{suffix}")
        for tool_name in server.tools:
            tool = tools[tool_name]
            safety = (
                "destructive"
                if tool.destructive
                else "readOnly"
                if tool.read_only
                else "approval required"
            )
            typer.echo(f"  - {tool.name} [{safety}]")


def _parse_test_command(value: str) -> TestCommand | None:
    """解析 `/test tool|skill|mcp <name> <request>`。"""

    parts = value.strip().split(maxsplit=3)
    if len(parts) != 4 or parts[0].lower() != "/test":
        return None
    kind = parts[1].lower()
    if kind not in {"tool", "skill", "mcp"}:
        return None
    target = parts[2].strip()
    prompt = parts[3].strip()
    if not target or not prompt:
        return None
    return TestCommand(kind=kind, target=target, prompt=prompt)


def _parse_search_mode(value: str) -> SearchMode | None:
    """解析 `/search-mode auto|rag|web|hybrid`。"""

    parts = value.strip().split()
    if len(parts) != 2 or parts[0].casefold() != "/search-mode":
        return None
    try:
        return SearchMode(parts[1].casefold())
    except ValueError:
        return None


def _build_test_prompt(command: TestCommand) -> str:
    """构建与 required_tool 配套的显式测试指令。"""

    if command.kind == "skill":
        return (
            f"这是 Skill 强制调用测试。必须先调用 load_skill，name 严格设置为 "
            f'"{command.target}"；加载成功后再执行：{command.prompt}。'
            "只有收到 ToolResult 才能声称 Skill 已加载。"
        )
    label = "MCP Tool" if command.kind == "mcp" else "Tool"
    return (
        f"这是 {label} 强制调用测试。必须调用 {command.target}；"
        f"测试要求：{command.prompt}。只有收到 ToolResult 才能声称工具已执行。"
    )


def _complete_run(
    client: AgentServerClient,
    conversation: ConversationInfo,
    result: dict[str, Any],
) -> None:
    """处理模型事件、Permission interrupt 和最终回答。

    Handle model events, permission interrupts, and the final answer.
    """

    conversation_id = conversation.conversation_id
    _show_model_events(result)
    while result.get("permission_request") is not None:
        _show_permission_request(result["permission_request"])
        approved = typer.confirm("Allow these tool calls?", default=False)
        result = client.resume_permission(
            conversation_id,
            str(result["run_id"]),
            approved,
        )
        _show_model_events(result)

    model = _selected_model(result, conversation.primary_model)
    _show_answer(
        result,
        conversation.agent_name,
        model,
        conversation.conversation_id,
    )


def _show_permission_request(request: Mapping[str, Any]) -> None:
    for item in request.get("requests", []):
        typer.echo(f"Tool: {item['tool_name']}")
        typer.echo(f"Input: {item['input']}")
        typer.echo(f"Reason: {item['reason']}")


def _show_model_events(result: Mapping[str, Any]) -> None:
    for event in result.get("model_events", []):
        _show_model_gateway_event(**event)


def _selected_model(result: Mapping[str, Any], default_model: str) -> str:
    """返回最终 Assistant 响应实际使用的模型。

    Return the model actually used for the final assistant response.
    """

    for event in reversed(result.get("model_events", [])):
        if event.get("event_type") == ModelGatewayEventType.SELECTED.value:
            return str(event["model"])
    return default_model


def _conversation_suffix(conversation_id: str) -> str:
    """返回用于 Chat 展示的 Conversation ID 后六位。

    Return the final six characters of a conversation ID for chat display.
    """

    return conversation_id[-6:]


def _command_argument(value: str, command: str) -> str | None:
    """读取 `/switch` 或 `/delete` 后的单个参数。

    Read the single argument following `/switch` or `/delete`.
    """

    parts = value.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != command:
        return None
    argument = parts[1].strip()
    return argument or None


def _resolve_conversation(
    reference: str,
    conversations: tuple[ConversationSummary, ...],
) -> ConversationSummary:
    """使用完整 ID 或唯一后缀解析 Conversation。

    Resolve a conversation from its full ID or a unique suffix.
    """

    exact = [item for item in conversations if item.conversation_id == reference]
    if exact:
        return exact[0]

    suffix_matches = [item for item in conversations if item.conversation_id.endswith(reference)]
    if not suffix_matches:
        raise ValueError(f"conversation not found: {reference}")
    if len(suffix_matches) > 1:
        raise ValueError(f"conversation suffix is ambiguous: {reference}")
    return suffix_matches[0]


def _show_conversations(
    conversations: tuple[ConversationSummary, ...],
    current_conversation_id: str,
) -> None:
    """列出当前用户的 Conversation 并标记当前项。

    List the current user's conversations and mark the active CLI selection.
    """

    if not conversations:
        typer.echo("No conversations.")
        return
    for item in conversations:
        marker = "*" if item.conversation_id == current_conversation_id else " "
        suffix = _conversation_suffix(item.conversation_id)
        typer.echo(f"{marker} [{suffix}] {item.status} {item.conversation_id}")


def _show_answer(
    result: Mapping[str, Any],
    agent_name: str,
    model: str,
    conversation_id: str,
) -> None:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            suffix = _conversation_suffix(conversation_id)
            typer.echo(f"{agent_name} [{suffix}] [{model}]: {message['content']}")
            return
    typer.echo("Agent did not return a text response.")


def main() -> None:
    """启动 Agent CLI。"""

    app()


__all__ = [
    "AgentServerClient",
    "ConversationDetail",
    "ConversationInfo",
    "ConversationSummary",
    "app",
    "chat",
    "index_command",
    "main",
    "root",
    "serve",
    "show_model_gateway_event",
]
