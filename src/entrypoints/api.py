"""单进程多用户 Agent Server。

Single-process multi-user agent server.
"""

from typing import Literal, NoReturn

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from entrypoints.bootstrap import AgentApplication, InvalidUserIdError, get_agent_application
from harness.conversation import (
    ConversationBusyError,
    ConversationForbiddenError,
    ConversationNotFoundError,
    ConversationRunCancelledError,
    ConversationRunResult,
    ConversationStatus,
    InvalidConversationInputError,
    RunNotFoundError,
)
from harness.error_recovery import OutputTokenRecoveryError, PromptTooLongRecoveryError
from harness.messages import Message
from harness.permissions import PermissionApproval, PermissionRequest
from services.mcp_tools import MCPToolAdapter
from services.model_gateway import (
    ModelGatewayEvent,
    ModelGatewayEventType,
    ModelGatewayRequestRejectedError,
    ModelGatewayUnavailableError,
)
from services.usage import TokenUsage

PUBLIC_RUN_RESPONSE_EXCLUDE = {
    "messages": {"__all__": {"provider_metadata"}},
}


class CreateConversationResponse(BaseModel):
    """创建 Conversation 的响应。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    status: ConversationStatus
    agent_name: str
    primary_model: str


class ConversationSummaryResponse(BaseModel):
    """Conversation 列表中的单条摘要。

    One summary item in a conversation list.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    status: ConversationStatus


class ListConversationsResponse(BaseModel):
    """当前用户拥有的 Conversation 列表。

    Conversations owned by the current user.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversations: tuple[ConversationSummaryResponse, ...]


class ConversationDetailResponse(BaseModel):
    """恢复已有 Conversation 所需的持久化元数据。

    Persisted metadata required to recover an existing conversation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    status: ConversationStatus
    active_run_id: str | None = None
    permission_request: PermissionRequest | None = None


class SendMessageRequest(BaseModel):
    """一条用户消息。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1, max_length=100_000)
    required_tool: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$",
    )


class ModelGatewayEventResponse(BaseModel):
    """可由客户端展示的模型重试或降级事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: ModelGatewayEventType
    model: str
    reason: str
    retry_number: int | None = None
    max_retries: int | None = None
    delay_seconds: float | None = None
    fallback_model: str | None = None

    @classmethod
    def from_event(cls, event: ModelGatewayEvent) -> "ModelGatewayEventResponse":
        return cls(
            event_type=event.event_type,
            model=event.model,
            reason=event.reason,
            retry_number=event.retry_number,
            max_retries=event.max_retries,
            delay_seconds=event.delay_seconds,
            fallback_model=event.fallback_model,
        )


class RunResponse(BaseModel):
    """一次 Run 的请求级结果和事件集合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    run_id: str
    status: ConversationStatus
    messages: tuple[Message, ...]
    permission_request: PermissionRequest | None = None
    model_events: tuple[ModelGatewayEventResponse, ...] = ()
    token_usage: TokenUsage = TokenUsage()

    @classmethod
    def from_result(
        cls,
        result: ConversationRunResult,
        events: list[ModelGatewayEvent],
        token_usage: TokenUsage,
    ) -> "RunResponse":
        return cls(
            conversation_id=result.conversation_id,
            run_id=result.run_id,
            status=result.status,
            messages=result.messages,
            permission_request=result.permission_request,
            model_events=tuple(ModelGatewayEventResponse.from_event(event) for event in events),
            token_usage=token_usage,
        )


class UserTokenUsageResponse(BaseModel):
    """当前用户在本进程内累计的真实模型 Token 用量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    token_usage: TokenUsage


class SkillSummaryResponse(BaseModel):
    """当前用户可加载的 Skill 元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str


class ListSkillsResponse(BaseModel):
    """当前用户可见的内置和私有 Skill。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skills: tuple[SkillSummaryResponse, ...]


class MCPToolSummaryResponse(BaseModel):
    """已发现 MCP Tool 的公开元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    server_name: str
    remote_name: str
    read_only: bool
    destructive: bool


class MCPServerSummaryResponse(BaseModel):
    """一个已配置 MCP Server 的发现状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Literal["connected", "failed"]
    tools: tuple[str, ...] = ()
    error_type: str | None = None


class MCPStatusResponse(BaseModel):
    """MCP 开关、Server 状态和已安装 Tool 清单。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    servers: tuple[MCPServerSummaryResponse, ...]
    tools: tuple[MCPToolSummaryResponse, ...]


def create_app(application: AgentApplication | None = None) -> FastAPI:
    """创建可注入测试依赖的 FastAPI 应用。

    Create a FastAPI application with injectable test dependencies.
    """

    server = FastAPI(title="Extensible Agent Server", version="0.1.0")

    def resolve_application() -> AgentApplication:
        return application or get_agent_application()

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def create_conversation(
        user_id: str,
    ) -> CreateConversationResponse:
        agent = resolve_application()
        try:
            record = agent.conversations.create(user_id)
        except InvalidUserIdError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return CreateConversationResponse(
            conversation_id=record.conversation_id,
            status=record.status,
            agent_name=agent.agent_name,
            primary_model=agent.primary_model,
        )

    async def list_conversations(user_id: str) -> ListConversationsResponse:
        agent = resolve_application()
        try:
            records = agent.conversations.list(user_id)
        except Exception as error:
            _raise_http_error(error)
        return ListConversationsResponse(
            conversations=tuple(
                ConversationSummaryResponse(
                    conversation_id=record.conversation_id,
                    status=record.status,
                )
                for record in records
            )
        )

    async def delete_conversation(
        user_id: str,
        conversation_id: str,
    ) -> Response:
        agent = resolve_application()
        try:
            await agent.conversations.delete(user_id, conversation_id)
        except Exception as error:
            _raise_http_error(error)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    async def get_conversation(
        user_id: str,
        conversation_id: str,
    ) -> ConversationDetailResponse:
        agent = resolve_application()
        try:
            record = agent.conversations.get(user_id, conversation_id)
        except Exception as error:
            _raise_http_error(error)
        return ConversationDetailResponse(
            conversation_id=record.conversation_id,
            status=record.status,
            active_run_id=record.active_run_id,
            permission_request=record.permission_request,
        )

    async def send_message(
        user_id: str,
        conversation_id: str,
        request: SendMessageRequest,
    ) -> RunResponse:
        agent = resolve_application()
        events: list[ModelGatewayEvent] = []
        usages: list[TokenUsage] = []
        try:
            with (
                agent.model_events.capture() as events,
                agent.model_usage.capture() as usages,
            ):
                result = await agent.conversations.send_message(
                    user_id,
                    conversation_id,
                    request.content,
                    request.required_tool,
                )
        except Exception as error:
            agent.usage_ledger.record(user_id, usages)
            _raise_http_error(error, events)
        request_usage = agent.usage_ledger.record(user_id, usages)
        return RunResponse.from_result(result, events, request_usage)

    async def resume_permission(
        user_id: str,
        conversation_id: str,
        run_id: str,
        approval: PermissionApproval,
    ) -> RunResponse:
        agent = resolve_application()
        events: list[ModelGatewayEvent] = []
        usages: list[TokenUsage] = []
        try:
            with (
                agent.model_events.capture() as events,
                agent.model_usage.capture() as usages,
            ):
                result = await agent.conversations.resume_permission(
                    user_id,
                    conversation_id,
                    run_id,
                    approval,
                )
        except Exception as error:
            agent.usage_ledger.record(user_id, usages)
            _raise_http_error(error, events)
        request_usage = agent.usage_ledger.record(user_id, usages)
        return RunResponse.from_result(result, events, request_usage)

    async def cancel_run(
        user_id: str,
        conversation_id: str,
        run_id: str,
    ) -> ConversationDetailResponse:
        """取消当前进程中正在执行的 Run，并释放 Conversation。"""

        agent = resolve_application()
        try:
            record = await agent.conversations.cancel_run(
                user_id,
                conversation_id,
                run_id,
            )
        except Exception as error:
            _raise_http_error(error)
        return ConversationDetailResponse(
            conversation_id=record.conversation_id,
            status=record.status,
            active_run_id=record.active_run_id,
            permission_request=record.permission_request,
        )

    async def get_token_usage(user_id: str) -> UserTokenUsageResponse:
        agent = resolve_application()
        try:
            agent.runtimes.get(user_id)
        except InvalidUserIdError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return UserTokenUsageResponse(
            user_id=user_id,
            token_usage=agent.usage_ledger.get(user_id),
        )

    async def list_skills(user_id: str) -> ListSkillsResponse:
        """返回当前用户 Runtime 实际安装的 Skill。"""

        agent = resolve_application()
        try:
            runtime = agent.runtimes.get(user_id)
        except InvalidUserIdError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return ListSkillsResponse(
            skills=tuple(
                SkillSummaryResponse(
                    name=manifest.name,
                    description=manifest.description,
                )
                for manifest in runtime.skill_manifests
            )
        )

    async def get_mcp_status(user_id: str) -> MCPStatusResponse:
        """返回服务端实际发现的 MCP Server 和 Tool 状态。"""

        agent = resolve_application()
        try:
            agent.runtimes.get(user_id)
        except InvalidUserIdError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        integration = agent.runtimes.mcp_integration
        tools = tuple(
            MCPToolSummaryResponse(
                name=tool.name,
                server_name=tool.server_name,
                remote_name=tool.remote_name,
                read_only=tool.read_only,
                destructive=tool.destructive,
            )
            for tool in integration.tools
            if isinstance(tool, MCPToolAdapter)
        )
        failures = {failure.server_name: failure for failure in integration.failures}
        servers = tuple(
            MCPServerSummaryResponse(
                name=server_name,
                status="failed" if server_name in failures else "connected",
                tools=tuple(
                    tool.name for tool in tools if tool.server_name == server_name
                ),
                error_type=(
                    failures[server_name].error_type
                    if server_name in failures
                    else None
                ),
            )
            for server_name in agent.runtimes.settings.mcp.servers
        )
        return MCPStatusResponse(
            enabled=agent.runtimes.settings.mcp.enabled,
            servers=servers,
            tools=tools,
        )

    server.add_api_route("/health", health, methods=["GET"])
    server.add_api_route(
        "/users/{user_id}/usage",
        get_token_usage,
        methods=["GET"],
        response_model=UserTokenUsageResponse,
    )
    server.add_api_route(
        "/users/{user_id}/skills",
        list_skills,
        methods=["GET"],
        response_model=ListSkillsResponse,
    )
    server.add_api_route(
        "/users/{user_id}/mcp",
        get_mcp_status,
        methods=["GET"],
        response_model=MCPStatusResponse,
    )
    server.add_api_route(
        "/users/{user_id}/conversations",
        create_conversation,
        methods=["POST"],
        response_model=CreateConversationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    server.add_api_route(
        "/users/{user_id}/conversations",
        list_conversations,
        methods=["GET"],
        response_model=ListConversationsResponse,
    )
    server.add_api_route(
        "/users/{user_id}/conversations/{conversation_id}",
        get_conversation,
        methods=["GET"],
        response_model=ConversationDetailResponse,
    )
    server.add_api_route(
        "/users/{user_id}/conversations/{conversation_id}",
        delete_conversation,
        methods=["DELETE"],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    server.add_api_route(
        "/users/{user_id}/conversations/{conversation_id}/messages",
        send_message,
        methods=["POST"],
        response_model=RunResponse,
        response_model_exclude=PUBLIC_RUN_RESPONSE_EXCLUDE,
    )
    server.add_api_route(
        "/users/{user_id}/conversations/{conversation_id}/runs/{run_id}/permission",
        resume_permission,
        methods=["POST"],
        response_model=RunResponse,
        response_model_exclude=PUBLIC_RUN_RESPONSE_EXCLUDE,
    )
    server.add_api_route(
        "/users/{user_id}/conversations/{conversation_id}/runs/{run_id}/cancel",
        cancel_run,
        methods=["POST"],
        response_model=ConversationDetailResponse,
    )
    return server


def _raise_http_error(
    error: Exception,
    model_events: list[ModelGatewayEvent] | None = None,
) -> NoReturn:
    """把 Conversation 领域错误映射成稳定 HTTP 状态码。"""

    if isinstance(error, ConversationForbiddenError):
        raise HTTPException(status_code=403, detail=str(error)) from error
    if isinstance(error, ConversationNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, ConversationBusyError | RunNotFoundError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, ConversationRunCancelledError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, InvalidUserIdError | InvalidConversationInputError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    if isinstance(error, ModelGatewayUnavailableError):
        visible_events = tuple(model_events or ())
        detail = {
            "message": str(error),
            "model_events": [
                ModelGatewayEventResponse.from_event(event).model_dump(mode="json")
                for event in visible_events
            ],
        }
        raise HTTPException(status_code=503, detail=detail) from error
    if isinstance(error, ModelGatewayRequestRejectedError):
        raise HTTPException(status_code=502, detail=str(error)) from error
    if isinstance(error, PromptTooLongRecoveryError):
        raise HTTPException(status_code=413, detail=str(error)) from error
    if isinstance(error, OutputTokenRecoveryError):
        raise HTTPException(status_code=502, detail=str(error)) from error
    raise error


app = create_app()


__all__ = [
    "ConversationSummaryResponse",
    "ConversationDetailResponse",
    "CreateConversationResponse",
    "ListConversationsResponse",
    "ListSkillsResponse",
    "MCPServerSummaryResponse",
    "MCPStatusResponse",
    "MCPToolSummaryResponse",
    "ModelGatewayEventResponse",
    "RunResponse",
    "SendMessageRequest",
    "SkillSummaryResponse",
    "UserTokenUsageResponse",
    "app",
    "create_app",
]
