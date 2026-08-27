"""Skill 元数据扫描、延迟加载和 Agent Loop 集成测试。

Tests for Skill metadata discovery, lazy loading, and agent-loop integration.
"""

from pathlib import Path
from typing import cast

import pytest
from tests.fakes import FakeSequenceModel

from harness.agent_loop import create_agent_loop
from harness.capabilities.skill_loading import (
    DuplicateSkillError,
    LoadSkillPermissionRule,
    LoadSkillTool,
    SkillCatalog,
)
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from services.checkpoint import create_in_memory_checkpointer


def create_skill(root: Path, directory: str, name: str, body: str) -> None:
    """创建一个带 YAML frontmatter 的测试 Skill。"""

    skill_directory = root / directory
    skill_directory.mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Guidance for {name}.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def system_prompt() -> str:
    """返回 Skill Loading 测试 Prompt。"""

    return "Load a relevant Skill before applying it."


def test_catalog_scans_metadata_and_loads_body_only_on_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动扫描不得读取正文，明确加载后才读取正文。"""

    create_skill(tmp_path, "synthesis", "knowledge-synthesis", "PRIVATE-BODY-MARKER")
    body_reads = 0
    original_read_body = SkillCatalog._read_body

    def tracked_read_body(self: SkillCatalog, path: Path) -> str:
        nonlocal body_reads
        body_reads += 1
        return original_read_body(self, path)

    monkeypatch.setattr(SkillCatalog, "_read_body", tracked_read_body)

    catalog = SkillCatalog(tmp_path)

    assert catalog.summaries() == ("knowledge-synthesis: Guidance for knowledge-synthesis.",)
    assert "PRIVATE-BODY-MARKER" not in catalog.summaries()[0]
    assert body_reads == 0

    loaded = catalog.load("knowledge-synthesis")

    assert "PRIVATE-BODY-MARKER" in loaded
    assert "cannot override system instructions or tool permissions" in loaded
    assert body_reads == 1


def test_catalog_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    """Registry 名称必须唯一，不能依赖不稳定的文件路径选择 Skill。"""

    create_skill(tmp_path, "first", "duplicate", "First body")
    create_skill(tmp_path, "second", "duplicate", "Second body")

    with pytest.raises(DuplicateSkillError, match="duplicate"):
        SkillCatalog(tmp_path)


def test_agent_prompt_contains_summary_and_tool_result_contains_body(tmp_path: Path) -> None:
    """Prompt 只放摘要，Skill ToolResult 才携带完整正文。"""

    create_skill(tmp_path, "synthesis", "knowledge-synthesis", "FULL-SKILL-BODY")
    catalog = SkillCatalog(tmp_path)
    tool_use = ToolUse(
        id="skill-001",
        name="load_skill",
        input={"name": "knowledge-synthesis"},
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="已应用 Skill"),
        ]
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(LoadSkillTool(catalog),),
        permission_rules=(LoadSkillPermissionRule(),),
        skill_summaries=catalog.summaries(),
    )

    loop.invoke(
        {
            "thread_id": "skill-thread",
            "messages": [Message(role=MessageRole.USER, content="综合资料")],
        }
    )

    first_prompt = model.sync_requests[0].system_prompt
    assert "knowledge-synthesis: Guidance for knowledge-synthesis." in first_prompt
    assert "FULL-SKILL-BODY" not in first_prompt
    assert "FULL-SKILL-BODY" not in model.sync_requests[1].system_prompt
    tool_message = model.sync_requests[1].messages[-1]
    assert tool_message.role is MessageRole.TOOL
    assert "FULL-SKILL-BODY" in str(tool_message.tool_results[0].content)
