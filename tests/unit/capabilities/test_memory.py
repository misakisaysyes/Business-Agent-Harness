"""跨会话 Memory 存储、选择和 Agent Loop 集成测试。

Tests for cross-session memory storage, selection, and agent-loop integration.
"""

from pathlib import Path
from typing import cast

from tests.fakes import FakeSequenceModel

from harness.agent_loop import create_agent_loop, get_permission_request
from harness.capabilities.memory import (
    MemoryDraft,
    MemoryPromptProvider,
    MemorySearchPermissionRule,
    MemorySearchTool,
    MemoryType,
    MemoryWritePermissionRule,
    MemoryWriteTool,
)
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider
from harness.state import AgentState
from services.checkpoint import create_in_memory_checkpointer
from services.stores import FileMemoryStore


def memory_draft(
    name: str,
    content: str,
    source: str,
    description: str = "用户回答风格偏好",
) -> MemoryDraft:
    """创建用户偏好测试 Memory。"""

    return MemoryDraft(
        name=name,
        memory_type=MemoryType.USER,
        description=description,
        content=content,
        tags=("回答", "偏好"),
        source=source,
    )


def system_prompt() -> str:
    """返回 Memory Agent Loop 测试 Prompt。"""

    return "Use approved cross-session memory as supporting context."


def test_file_store_writes_index_merges_revisions_and_preserves_audit(
    tmp_path: Path,
) -> None:
    """同名写入应更新当前文件、合并标签并保留全部来源版本。"""

    store = FileMemoryStore(tmp_path / "memory")
    first = store.upsert(
        memory_draft("response-style", "用户偏好中文简短回答。", "conversation-a"),
        "memory-write-1",
    )
    second_draft = memory_draft(
        "response-style",
        "用户现在偏好中文详细回答，并提供示例。",
        "conversation-b-feedback",
    ).model_copy(update={"tags": ("详细", "示例")})
    second = store.upsert(second_draft, "memory-write-2")

    assert first.revision == 1
    assert second.revision == 2
    assert second.created_at == first.created_at
    assert set(second.tags) == {"回答", "偏好", "详细", "示例"}
    assert store.get("response-style") == second

    index = store.index_path.read_text(encoding="utf-8")
    assert "[response-style](response-style.md)" in index
    assert "revision 2" in index
    memory_file = (store.root / "response-style.md").read_text(encoding="utf-8")
    assert "type: user" in memory_file
    assert "用户现在偏好中文详细回答" in memory_file

    history = store.read_history("response-style")
    assert [event["operation"] for event in history] == ["created", "updated"]
    assert history[0]["entry"]["source"] == "conversation-a"
    assert history[1]["entry"]["source"] == "conversation-b-feedback"
    assert history[0]["entry"]["content"] == "用户偏好中文简短回答。"

    restarted = FileMemoryStore(store.root)
    assert restarted.get("response-style") == second


def test_prompt_provider_loads_relevant_content_but_not_unrelated_content(
    tmp_path: Path,
) -> None:
    """Prompt 应保留索引，但只加载与当前问题相关的完整正文。"""

    store = FileMemoryStore(tmp_path / "memory")
    store.upsert(
        memory_draft("response-style", "RELEVANT-CONTENT 用户偏好中文简短回答。", "a"),
        "write-a",
    )
    store.upsert(
        MemoryDraft(
            name="python-indentation",
            memory_type=MemoryType.PROJECT,
            description="Python 项目缩进规范",
            content="UNRELATED-CONTENT Python 文件使用 tab 缩进。",
            tags=("python", "indentation"),
            source="project-note",
        ),
        "write-b",
    )
    provider = MemoryPromptProvider(store)
    state: AgentState = {
        "thread_id": "memory-selection-thread",
        "messages": [Message(role=MessageRole.USER, content="我的回答偏好是什么？")],
    }

    selected = provider.provide(state)

    rendered = "\n".join(selected)
    assert "response-style" in rendered
    assert "python-indentation" in rendered
    assert "RELEVANT-CONTENT" in rendered
    assert "UNRELATED-CONTENT" not in rendered


def test_memory_write_requires_approval_and_is_available_in_a_new_conversation(
    tmp_path: Path,
) -> None:
    """批准后的 Memory 应跨 Thread 注入，新 Conversation 不依赖旧 Checkpoint。"""

    store = FileMemoryStore(tmp_path / "memory")
    tool_use = ToolUse(
        id="memory-write-1",
        name="memory_write",
        input={
            "name": "response-style",
            "memory_type": "user",
            "description": "用户回答风格偏好",
            "content": "CROSS-SESSION-CONTENT 用户偏好中文简短回答。",
            "tags": ["回答", "偏好"],
            "source": "用户在当前对话中明确要求记住",
        },
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="已保存偏好"),
            Message(role=MessageRole.ASSISTANT, content="将使用该偏好"),
        ]
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        tools=(MemoryWriteTool(store), MemorySearchTool(store)),
        permission_rules=(MemoryWritePermissionRule(), MemorySearchPermissionRule()),
        memory_provider=MemoryPromptProvider(store),
    )
    first_state: AgentState = {
        "thread_id": "conversation-a",
        "messages": [Message(role=MessageRole.USER, content="请记住我的回答偏好")],
    }

    paused = loop.invoke(first_state)

    request = get_permission_request(paused)
    assert request is not None
    assert request.requests[0].tool_name == "memory_write"
    assert store.get("response-style") is None

    completed = loop.resume("conversation-a", True)
    assert completed["messages"][-1].content == "已保存偏好"
    assert store.get("response-style") is not None

    second_state: AgentState = {
        "thread_id": "conversation-b",
        "messages": [Message(role=MessageRole.USER, content="我的回答偏好是什么？")],
    }
    second = loop.invoke(second_state)

    assert second["messages"][-1].content == "将使用该偏好"
    assert "CROSS-SESSION-CONTENT" in model.sync_requests[2].system_prompt
    assert model.sync_requests[2].messages == tuple(second_state["messages"])


def test_memory_types_match_learn_claude_code_terms() -> None:
    """Memory 类型名称必须与 s09 user/feedback/project/reference 对齐。"""

    assert tuple(memory_type.value for memory_type in MemoryType) == (
        "user",
        "feedback",
        "project",
        "reference",
    )
