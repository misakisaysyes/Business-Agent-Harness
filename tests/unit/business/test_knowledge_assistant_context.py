"""Knowledge Assistant 业务 Context 测试。

Knowledge Assistant business-context tests.
"""

from pathlib import Path

from business.knowledge_assistant.context import KnowledgeAssistantContextProvider
from harness.messages import Message, MessageRole
from harness.state import AgentState


def test_business_context_exposes_task_and_safe_relative_material_names(
    tmp_path: Path,
) -> None:
    """业务 Context 不得读取正文、隐藏文件或暴露绝对目录。"""

    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "guides").mkdir(parents=True)
    (knowledge_root / "guides/intro.md").write_text("PRIVATE CONTENT", encoding="utf-8")
    (knowledge_root / ".env").write_text("API_KEY=secret", encoding="utf-8")
    state: AgentState = {
        "thread_id": "business-context",
        "messages": [Message(role=MessageRole.USER, content="总结可用资料")],
    }

    fragments = KnowledgeAssistantContextProvider(knowledge_root).provide(state)
    rendered = "\n".join(fragment.content for fragment in fragments)

    assert fragments[0].title == "Current Task"
    assert fragments[0].content == "总结可用资料"
    assert "guides/intro.md" in rendered
    assert "PRIVATE CONTENT" not in rendered
    assert "API_KEY" not in rendered
    assert str(tmp_path) not in rendered
