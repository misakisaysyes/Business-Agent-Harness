"""Knowledge Assistant 的能力和依赖装配。

Knowledge Assistant capability and dependency composition.
"""

from pathlib import Path

from business.knowledge_assistant.context import KnowledgeAssistantContextProvider
from business.knowledge_assistant.permission_rules import (
    CalculatorPermissionRule,
    ExternalPublishPermissionRule,
    FileReadPermissionRule,
    ReportWritePermissionRule,
)
from business.knowledge_assistant.system_prompt import get_system_prompt
from business.knowledge_assistant.tools import CalculatorTool, FileReaderTool, ReportWriterTool
from harness.capabilities.memory import (
    MemoryPromptProvider,
    MemorySearchPermissionRule,
    MemorySearchTool,
    MemorySelectionConfig,
    MemoryStore,
    MemoryWritePermissionRule,
    MemoryWriteTool,
)
from harness.capabilities.skill_loading import (
    LoadSkillPermissionRule,
    LoadSkillTool,
    SkillCatalog,
)
from harness.capabilities.task_system import (
    InMemoryTaskStore,
    TaskStore,
    TaskSystemPermissionRule,
    create_task_tools,
)
from harness.capabilities.todo_write import TodoWritePermissionRule, TodoWriteTool
from harness.hooks import (
    LargeOutputWarningHook,
    PermissionCheckHook,
    StopMetricsHook,
    ToolCallLoggingHook,
)
from harness.profile import AgentProfile, Capability, ModelConfigRef
from services.artifacts import ArtifactStore

BUSINESS_AGENT_NAME = "knowledge_assistant"


def create_skill_catalog(
    skills_root: str | Path | None = None,
    private_skills_root: str | Path | None = None,
) -> SkillCatalog:
    """创建包含内置和当前用户私有 Skill 的目录。

    Create a catalog containing built-in and current-user private skills.
    """

    builtin_skills = (
        Path(skills_root).resolve()
        if skills_root is not None
        else Path(__file__).parent / "skills"
    )
    skill_roots = [builtin_skills]
    if private_skills_root is not None:
        skill_roots.append(Path(private_skills_root).resolve())
    return SkillCatalog(skill_roots)


def create_knowledge_assistant_profile(
    workspace_root: str | Path,
    knowledge_root: str | Path,
    artifact_root: str | Path,
    skills_root: str | Path | None = None,
    private_knowledge_root: str | Path | None = None,
    private_skills_root: str | Path | None = None,
    memory_store: MemoryStore | None = None,
    memory_config: MemorySelectionConfig | None = None,
    task_store: TaskStore | None = None,
    skill_catalog: SkillCatalog | None = None,
) -> AgentProfile:
    """使用运行目录创建 Knowledge Assistant Profile。

    Create the Knowledge Assistant profile from runtime directories.
    """

    workspace = Path(workspace_root).resolve()
    knowledge = Path(knowledge_root).resolve()
    artifacts = Path(artifact_root).resolve()
    knowledge_roots = [knowledge]
    if private_knowledge_root is not None:
        knowledge_roots.insert(0, Path(private_knowledge_root).resolve())
    file_reader = FileReaderTool(
        (workspace, *knowledge_roots, artifacts),
        default_root=knowledge_roots[0],
    )
    artifact_store = ArtifactStore(artifacts)
    active_skill_catalog = skill_catalog or create_skill_catalog(
        skills_root,
        private_skills_root,
    )
    memory_tools = (
        (MemoryWriteTool(memory_store), MemorySearchTool(memory_store))
        if memory_store is not None
        else ()
    )
    memory_rules = (
        (MemoryWritePermissionRule(), MemorySearchPermissionRule())
        if memory_store is not None
        else ()
    )
    active_task_store = task_store or InMemoryTaskStore()

    return AgentProfile(
        name=BUSINESS_AGENT_NAME,
        model=ModelConfigRef(name="default"),
        system_prompt=get_system_prompt,
        tools=(
            CalculatorTool(),
            file_reader,
            ReportWriterTool(artifact_store),
            TodoWriteTool(),
            LoadSkillTool(active_skill_catalog),
            *create_task_tools(active_task_store),
            *memory_tools,
        ),
        permission_rules=(
            CalculatorPermissionRule(),
            FileReadPermissionRule(
                file_reader,
                auto_allowed_roots=(*knowledge_roots, artifacts),
            ),
            ReportWritePermissionRule(artifact_store),
            ExternalPublishPermissionRule(),
            TodoWritePermissionRule(),
            LoadSkillPermissionRule(),
            TaskSystemPermissionRule(),
            *memory_rules,
        ),
        hooks=(
            PermissionCheckHook(),
            ToolCallLoggingHook(),
            LargeOutputWarningHook(),
            StopMetricsHook(),
        ),
        context_providers=(KnowledgeAssistantContextProvider(knowledge_roots),),
        skill_summaries=active_skill_catalog.summaries(),
        memory_provider=(
            MemoryPromptProvider(memory_store, memory_config)
            if memory_store is not None
            else None
        ),
        capabilities=(
            Capability(name="todo_write"),
            Capability(name="skill_loading"),
            Capability(name="context_compact"),
            Capability(name="task_system"),
            *((Capability(name="memory"),) if memory_store is not None else ()),
        ),
    )


__all__ = [
    "BUSINESS_AGENT_NAME",
    "create_knowledge_assistant_profile",
    "create_skill_catalog",
]
