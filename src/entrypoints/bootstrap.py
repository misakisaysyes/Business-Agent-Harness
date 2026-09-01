"""Agent Loop 和共享服务的启动装配。

Agent loop and shared-service bootstrap.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from business.knowledge_assistant import (
    BUSINESS_AGENT_NAME,
    create_knowledge_assistant_profile,
)
from business.knowledge_assistant.agent_teams.analyst import build_analyst_definition
from business.knowledge_assistant.agent_teams.lead import (
    DelegateAnalysisTool,
    DelegateResearchTool,
    RequestReviewTool,
)
from business.knowledge_assistant.agent_teams.researcher import build_researcher_definitions
from business.knowledge_assistant.agent_teams.reviewer import build_reviewer_definition
from business.knowledge_assistant.permission_rules import (
    DelegateAnalysisPermissionRule,
    DelegateResearchPermissionRule,
    RequestReviewPermissionRule,
)
from business.knowledge_assistant.profile import create_skill_catalog
from harness.agent_loop import AgentLoop, create_agent_loop
from harness.capabilities.agent_teams.contracts import (
    DelegationBudget,
    SubagentContext,
    SubagentDefinition,
)
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.context_compact import ContextCompactConfig, ContextCompactor
from harness.capabilities.memory import MemorySelectionConfig
from harness.capabilities.rag import AccessScope, RAGPipeline
from harness.capabilities.skill_loading import SkillCatalog, SkillManifest
from harness.capabilities.subagent import SubagentRunner
from harness.capabilities.task_system import TaskStore
from harness.conversation import ConversationService
from harness.error_recovery import ErrorRecoveryPolicy
from harness.model import ModelProvider
from harness.profile import Capability, ModelConfigRef
from services.artifacts import ArtifactStore
from services.checkpoint import create_sqlite_checkpointer
from services.config import AppSettings, ModelSettings, get_settings
from services.mcp_tools import (
    MCPIntegration,
    MCPServerFailure,
    load_mcp_integration_sync,
)
from services.model_gateway import (
    ModelGateway,
    ModelGatewayEventHandler,
    ModelRoute,
)
from services.observability import ModelGatewayEventCollector
from services.rag import create_rag_components
from services.stores import FileMemoryStore, SQLiteConversationStore, SQLiteTaskStore
from services.usage import ModelUsageCollector, UsageTrackingModel, UserTokenUsageLedger

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class UnknownModelConfigError(LookupError):
    """Profile 引用了不存在的模型配置。

    Raised when a profile references an unknown model configuration.
    """


class InvalidUserIdError(ValueError):
    """本地用户标识不符合路径安全规则。

    Raised when a local user ID violates path-safety rules.
    """


@dataclass(frozen=True, slots=True)
class UserRuntime:
    """一个可信本地用户复用的 Agent Runtime。

    Agent runtime reused by one trusted local user.
    """

    user_id: str
    agent_loop: AgentLoop
    workspace_root: Path
    private_knowledge_root: Path
    private_skills_root: Path
    artifact_root: Path
    memory_root: Path
    checkpoint_path: Path
    task_database_path: Path
    skill_manifests: tuple[SkillManifest, ...]


@dataclass(frozen=True, slots=True)
class AgentApplication:
    """Agent Server 需要的进程级共享对象。

    Process-wide objects required by the agent server.
    """

    runtimes: "UserRuntimeRegistry"
    conversations: ConversationService
    model_events: ModelGatewayEventCollector
    model_usage: ModelUsageCollector
    usage_ledger: UserTokenUsageLedger
    agent_name: str
    primary_model: str
    model_choices: tuple[str, ...]
    mcp_failures: tuple[MCPServerFailure, ...]


def resolve_model_settings(
    reference: ModelConfigRef,
    settings: AppSettings,
) -> ModelSettings:
    """根据 Profile 引用解析具体模型配置。

    Resolve concrete model settings from a profile reference.
    """

    if reference.name != "default":
        raise UnknownModelConfigError(f"unknown model configuration: {reference.name}")
    return settings.model


def resolve_runtime_path(path: Path, workspace_root: Path) -> Path:
    """相对路径以 Workspace 为基准解析。

    Resolve relative runtime paths against the workspace root.
    """

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (workspace_root / expanded).resolve()


def create_shared_model(
    settings: AppSettings,
    model_event_handler: ModelGatewayEventHandler | None = None,
) -> ModelProvider:
    """创建供所有用户 Runtime 共享的模型或 ModelGateway。

    Create the model or model gateway shared by every user runtime.
    """

    from services.models import create_model_provider

    model_settings = settings.model
    primary_provider = create_model_provider(model_settings)
    if not settings.model_gateway.enabled:
        return primary_provider

    fallback_routes: tuple[ModelRoute, ...] = ()
    if settings.fallback_model is not None:
        fallback_settings = settings.fallback_model
        fallback_provider = create_model_provider(fallback_settings)
        fallback_routes = (_create_model_route(fallback_provider, fallback_settings),)

    return ModelGateway(
        primary=_create_model_route(primary_provider, model_settings),
        fallbacks=fallback_routes,
        settings=settings.model_gateway,
        event_handler=model_event_handler,
    )


def bootstrap_agent(
    model: ModelProvider | None = None,
    settings: AppSettings | None = None,
    model_event_handler: ModelGatewayEventHandler | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    artifact_root: Path | None = None,
    workspace_root: Path | None = None,
    private_knowledge_root: Path | None = None,
    private_skills_root: Path | None = None,
    memory_root: Path | None = None,
    checkpoint_path: Path | None = None,
    task_store: TaskStore | None = None,
    task_database_path: Path | None = None,
    mcp_integration: MCPIntegration | None = None,
    skill_catalog: SkillCatalog | None = None,
    rag_pipeline: RAGPipeline | None = None,
    trusted_user_id: str | None = None,
) -> AgentLoop:
    """解析 Profile 和 Model，并创建可运行的 AgentLoop。

    Resolve the profile and model, then create a runnable AgentLoop.
    """

    active_settings = settings or get_settings()
    configured_workspace = active_settings.paths.workspace_root.expanduser().resolve()
    active_workspace = workspace_root or configured_workspace
    active_memory_root = memory_root or (active_workspace / ".memory").resolve()
    memory_config = MemorySelectionConfig.model_validate(
        active_settings.memory.model_dump(
            exclude={"enabled", "max_memories"},
        )
    )
    memory_store = (
        FileMemoryStore(
            active_memory_root,
            max_memories=active_settings.memory.max_memories,
        )
        if active_settings.memory.enabled
        else None
    )
    active_task_store = task_store or SQLiteTaskStore(
        task_database_path or active_workspace / ".agent/tasks.sqlite3",
        busy_timeout_seconds=active_settings.task_system.busy_timeout_seconds,
    )

    active_rag_pipeline = rag_pipeline
    if active_settings.rag.enabled and active_rag_pipeline is None:
        active_rag_pipeline = create_rag_components(active_settings.rag).pipeline

    profile = create_knowledge_assistant_profile(
        workspace_root=active_workspace,
        knowledge_root=resolve_runtime_path(
            active_settings.paths.knowledge_root,
            configured_workspace,
        ),
        artifact_root=artifact_root
        or resolve_runtime_path(active_settings.paths.artifact_root, configured_workspace),
        private_knowledge_root=private_knowledge_root,
        private_skills_root=private_skills_root,
        memory_store=memory_store,
        memory_config=memory_config,
        task_store=active_task_store,
        skill_catalog=skill_catalog,
        rag_pipeline=active_rag_pipeline,
        rag_access_scope=(
            AccessScope(user_id=trusted_user_id, include_public=True)
            if active_rag_pipeline is not None
            else None
        ),
        rag_top_k=active_settings.rag.top_k,
        rag_score_threshold=active_settings.rag.score_threshold,
    )
    active_mcp = (
        mcp_integration
        if mcp_integration is not None
        else load_mcp_integration_sync(
            active_settings.mcp,
            reserved_names=tuple(tool.name for tool in profile.tools),
        )
    )
    if active_mcp.tools:
        profile = profile.model_copy(
            update={
                "tools": (*profile.tools, *active_mcp.tools),
                "permission_rules": (
                    *profile.permission_rules,
                    *active_mcp.permission_rules,
                ),
                "capabilities": (*profile.capabilities, Capability(name="mcp_tools")),
            }
        )

    resolve_model_settings(profile.model, active_settings)
    active_model = model or create_shared_model(active_settings, model_event_handler)
    active_artifact_root = artifact_root or resolve_runtime_path(
        active_settings.paths.artifact_root,
        configured_workspace,
    )
    compact_config = ContextCompactConfig.model_validate(
        active_settings.context_compact.model_dump()
    )
    recovery_policy = ErrorRecoveryPolicy(
        **active_settings.error_recovery.model_dump()
    )
    context_compactor = (
        ContextCompactor(
            active_model,
            ArtifactStore(active_artifact_root),
            compact_config,
        )
        if compact_config.enabled
        else None
    )

    active_checkpointer = checkpointer or create_sqlite_checkpointer(
        checkpoint_path or active_workspace / ".agent/checkpoints.sqlite3",
        busy_timeout_seconds=active_settings.checkpoint.busy_timeout_seconds,
    )

    # M7: Researcher 是按任务创建的隔离 Agent Loop。子 Loop 只拿到角色
    # allowlist 中的工具，因此不会递归拿到 Lead 的 delegate_research。
    base_profile = profile
    available_tool_names = tuple(tool.name for tool in base_profile.tools)
    researcher_definitions = build_researcher_definitions(
        available_tool_names,
        max_iterations=active_settings.agent_loop.max_iterations,
    )
    team_definitions = {
        **researcher_definitions,
        "analyst": build_analyst_definition(
            available_tool_names,
            max_iterations=active_settings.agent_loop.max_iterations,
        ),
        "reviewer": build_reviewer_definition(
            max_iterations=active_settings.agent_loop.max_iterations,
        ),
    }

    def create_researcher_loop(
        definition: SubagentDefinition,
        context: SubagentContext,
    ) -> AgentLoop:
        allowed_names = frozenset(
            context.allowed_tool_names or definition.allowed_tool_names
        )
        selected_tools = tuple(
            tool for tool in base_profile.tools if tool.name in allowed_names
        )
        if not selected_tools and definition.role.endswith("researcher"):
            raise RuntimeError(
                f"no configured tools are available for researcher role: {definition.role}"
            )
        selected_rules = tuple(
            rule
            for rule in base_profile.permission_rules
            if getattr(rule, "name", "")
            in {
                "allow_document_catalog",
                "allow_document_search",
                "allow_calculator",
                "search_mode_policy",
                "mcp_tool_annotations",
            }
        )
        return create_agent_loop(
            active_model,
            lambda: definition.system_prompt,
            checkpointer=active_checkpointer,
            tools=selected_tools,
            permission_rules=selected_rules,
            hooks=base_profile.hooks,
            hook_failure_mode=base_profile.hook_failure_mode,
            context_providers=base_profile.context_providers,
            skill_summaries=(),
            context_compactor=context_compactor,
            error_recovery=recovery_policy,
            max_iterations=min(
                definition.max_iterations,
                active_settings.agent_loop.max_iterations,
            ),
        )

    coordinator = TeamCoordinator(
        SubagentRunner(
            create_researcher_loop,
            max_context_chars=DelegationBudget().max_context_chars,
        ),
        team_definitions,
    )
    delegate_research = DelegateResearchTool(coordinator)
    delegate_analysis = DelegateAnalysisTool(coordinator)
    request_review = RequestReviewTool(coordinator)
    profile = base_profile.model_copy(
        update={
            "tools": (
                *base_profile.tools,
                delegate_research,
                delegate_analysis,
                request_review,
            ),
            "permission_rules": (
                *base_profile.permission_rules,
                DelegateResearchPermissionRule(),
                DelegateAnalysisPermissionRule(),
                RequestReviewPermissionRule(),
            ),
            "capabilities": (
                *base_profile.capabilities,
                Capability(name="multi_agent"),
            ),
        }
    )

    return create_agent_loop(
        active_model,
        profile.system_prompt,
        checkpointer=active_checkpointer,
        tools=profile.tools,
        permission_rules=profile.permission_rules,
        hooks=profile.hooks,
        hook_failure_mode=profile.hook_failure_mode,
        context_providers=profile.context_providers,
        skill_summaries=profile.skill_summaries,
        memory_provider=profile.memory_provider,
        context_compactor=context_compactor,
        error_recovery=recovery_policy,
        max_iterations=active_settings.agent_loop.max_iterations,
    )


class UserRuntimeRegistry:
    """按用户懒加载并复用 AgentLoop 和 Checkpointer。

    Lazily create and reuse one agent loop and checkpointer per user.
    """

    def __init__(
        self,
        model: ModelProvider,
        settings: AppSettings,
        mcp_integration: MCPIntegration | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.mcp_integration = (
            mcp_integration
            if mcp_integration is not None
            else load_mcp_integration_sync(settings.mcp)
        )
        self._runtimes: dict[str, UserRuntime] = {}
        self._rag_pipeline = (
            create_rag_components(settings.rag).pipeline if settings.rag.enabled else None
        )

        workspace_root = settings.paths.workspace_root.expanduser().resolve()
        self._artifact_root = resolve_runtime_path(
            settings.paths.artifact_root,
            workspace_root,
        )
        self._user_data_root = resolve_runtime_path(
            settings.paths.user_data_root,
            workspace_root,
        )

    def get(self, user_id: str) -> UserRuntime:
        """返回用户 Runtime；首次调用时创建。

        Return a user's runtime, creating it on first access.
        """

        self._validate_user_id(user_id)
        runtime = self._runtimes.get(user_id)
        if runtime is not None:
            return runtime

        artifact_root = (self._artifact_root / user_id).resolve()
        if not artifact_root.is_relative_to(self._artifact_root):
            raise InvalidUserIdError("user ID escapes the artifact root")

        user_root = (self._user_data_root / user_id).resolve()
        if not user_root.is_relative_to(self._user_data_root):
            raise InvalidUserIdError("user ID escapes the user-data root")
        self._user_data_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        user_root.mkdir(mode=0o700, exist_ok=True)
        workspace_root = user_root / "workspace"
        private_knowledge_root = user_root / "knowledge"
        private_skills_root = user_root / "skills"
        memory_root = user_root / "memory"
        checkpoint_path = user_root / self.settings.checkpoint.user_database_filename
        task_database_path = user_root / self.settings.task_system.database_filename
        for path in (
            workspace_root,
            private_knowledge_root,
            private_skills_root,
            memory_root,
            artifact_root,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)

        skill_catalog = create_skill_catalog(
            private_skills_root=private_skills_root,
        )

        runtime = UserRuntime(
            user_id=user_id,
            agent_loop=bootstrap_agent(
                model=self.model,
                settings=self.settings,
                artifact_root=artifact_root,
                workspace_root=workspace_root,
                private_knowledge_root=private_knowledge_root,
                private_skills_root=private_skills_root,
                memory_root=memory_root,
                checkpoint_path=checkpoint_path,
                task_database_path=task_database_path,
                mcp_integration=self.mcp_integration,
                skill_catalog=skill_catalog,
                rag_pipeline=self._rag_pipeline,
                trusted_user_id=user_id,
            ),
            workspace_root=workspace_root,
            private_knowledge_root=private_knowledge_root,
            private_skills_root=private_skills_root,
            artifact_root=artifact_root,
            memory_root=memory_root,
            checkpoint_path=checkpoint_path,
            task_database_path=task_database_path,
            skill_manifests=skill_catalog.manifests,
        )
        self._runtimes[user_id] = runtime
        return runtime

    def get_agent_loop(self, user_id: str) -> AgentLoop:
        """返回 ConversationService 需要的用户 AgentLoop。"""

        return self.get(user_id).agent_loop

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not USER_ID_PATTERN.fullmatch(user_id):
            raise InvalidUserIdError(
                "user ID must contain 1-64 letters, numbers, underscores, or hyphens"
            )


def create_agent_application(
    model: ModelProvider | None = None,
    settings: AppSettings | None = None,
    mcp_integration: MCPIntegration | None = None,
) -> AgentApplication:
    """创建单进程 Agent Server 的共享应用对象。

    Create the shared application objects for the single-process agent server.
    """

    active_settings = settings or get_settings()
    model_events = ModelGatewayEventCollector()
    base_model = model or create_shared_model(active_settings, model_events.emit)
    model_usage = ModelUsageCollector()
    shared_model = UsageTrackingModel(base_model, model_usage)
    usage_ledger = UserTokenUsageLedger()
    runtimes = UserRuntimeRegistry(shared_model, active_settings, mcp_integration)
    workspace_root = active_settings.paths.workspace_root.expanduser().resolve()
    conversation_database_path = resolve_runtime_path(
        active_settings.checkpoint.conversation_database_path,
        workspace_root,
    )
    conversation_store = SQLiteConversationStore(
        conversation_database_path,
        busy_timeout_seconds=active_settings.checkpoint.busy_timeout_seconds,
    )
    if isinstance(base_model, ModelGateway):
        model_choices = tuple(route.label for route in base_model.routes)
        primary_model = model_choices[0]
    else:
        configured_model = active_settings.model
        primary_model = (
            f"{configured_model.provider}/{configured_model.model_id}"
            if configured_model.provider is not None and configured_model.model_id is not None
            else base_model.name
        )
        model_choices = (primary_model,)
    return AgentApplication(
        runtimes=runtimes,
        conversations=ConversationService(runtimes.get_agent_loop, conversation_store),
        model_events=model_events,
        model_usage=model_usage,
        usage_ledger=usage_ledger,
        agent_name=BUSINESS_AGENT_NAME,
        primary_model=primary_model,
        model_choices=model_choices,
        mcp_failures=runtimes.mcp_integration.failures,
    )


@lru_cache(maxsize=1)
def get_agent_application() -> AgentApplication:
    """返回当前 Server 进程复用的 AgentApplication。"""

    return create_agent_application()


def _create_model_route(
    provider: ModelProvider,
    settings: ModelSettings,
) -> ModelRoute:
    """从已校验配置创建一个模型网关路由。

    Create one model-gateway route from validated settings.
    """

    if settings.model_id is None:
        raise ValueError("model ID is required for a gateway route")
    return ModelRoute(
        provider=provider,
        model_id=settings.model_id,
        quota_scope=settings.quota_scope or provider.name,
        max_concurrency=settings.max_concurrency,
        # 当前 DeepSeek V4 Thinking 会拒绝 required/specified tool_choice。
        # DeepSeek V4 thinking currently rejects required/specified tool_choice.
        supports_required_tool_choice=settings.provider.lower() not in {"deepseek", "ds"}
        if settings.provider is not None
        else True,
    )


__all__ = [
    "AgentApplication",
    "InvalidUserIdError",
    "UnknownModelConfigError",
    "UserRuntime",
    "UserRuntimeRegistry",
    "bootstrap_agent",
    "create_agent_application",
    "create_shared_model",
    "get_agent_application",
    "resolve_model_settings",
    "resolve_runtime_path",
]
