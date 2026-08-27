"""Knowledge Assistant Profile 装配测试。

Tests for Knowledge Assistant profile composition.
"""

import inspect
from pathlib import Path

import business.knowledge_assistant.profile as profile_module
from business.knowledge_assistant import create_knowledge_assistant_profile
from business.knowledge_assistant.system_prompt import get_system_prompt
from business.knowledge_assistant.tools import FileReaderTool
from services.stores import FileMemoryStore


def test_knowledge_assistant_profile_uses_shared_contracts(tmp_path: Path) -> None:
    """业务 Profile 应只组合 Harness 契约。

    The business profile should compose only harness contracts.
    """

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    artifacts = workspace / "artifacts"
    memory_store = FileMemoryStore(tmp_path / "memory")
    profile = create_knowledge_assistant_profile(
        workspace,
        knowledge,
        artifacts,
        memory_store=memory_store,
    )

    assert profile.name == "knowledge_assistant"
    assert profile.model.name == "default"
    assert profile.system_prompt().strip()
    assert tuple(tool.name for tool in profile.tools) == (
        "calculator",
        "file_reader",
        "report_writer",
        "todo_write",
        "load_skill",
        "create_task",
        "get_task",
        "list_tasks",
        "claim_task",
        "complete_task",
        "fail_task",
        "memory_write",
        "memory_search",
    )
    assert tuple(rule.name for rule in profile.permission_rules) == (
        "allow_calculator",
        "authorized_file_read",
        "report_write",
        "deny_external_publish",
        "allow_todo_write",
        "allow_load_skill",
        "allow_task_system",
        "confirm_memory_write",
        "allow_memory_search",
    )
    assert tuple(hook.name for hook in profile.hooks) == (
        "permission_check",
        "tool_call_logging",
        "large_output_warning",
        "stop_metrics",
    )
    assert tuple(provider.name for provider in profile.context_providers) == (
        "knowledge_assistant_context",
    )
    assert profile.skill_summaries == (
        "knowledge-synthesis: Synthesize facts from authorized local materials and clearly "
        "separate evidence, inference, and missing information.",
    )
    assert tuple(capability.name for capability in profile.capabilities) == (
        "todo_write",
        "skill_loading",
        "context_compact",
        "task_system",
        "memory",
    )
    assert profile.memory_provider is not None

    reader = next(tool for tool in profile.tools if tool.name == "file_reader")
    assert isinstance(reader, FileReaderTool)
    assert reader.allowed_roots == (
        workspace.resolve(),
        knowledge.resolve(),
        artifacts.resolve(),
    )
    assert reader.default_root == knowledge.resolve()


def test_business_profile_does_not_import_model_services() -> None:
    """业务 Profile 不得直接依赖具体模型适配器。

    The business profile must not depend directly on concrete model adapters.
    """

    source = inspect.getsource(profile_module)

    assert "services.models" not in source
    assert "ChatAnthropic" not in source


def test_second_phase_prompt_allows_rag_but_keeps_agent_teams_disabled() -> None:
    """第二阶段 Prompt 不得同时要求并禁止 document_search。"""

    prompt = get_system_prompt()

    assert "When document_search is available" in prompt
    assert "Do not use RAG/document_search" not in prompt
    assert "Do not use subagents or agent teams" in prompt
