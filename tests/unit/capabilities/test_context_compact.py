"""Context Compact 分层压缩测试。

Tests for layered context compaction.
"""

import json
from pathlib import Path
from typing import cast

import pytest
from tests.fakes import FakeModel, FakeSequenceModel

from harness.agent_loop import create_agent_loop
from harness.capabilities.context_compact import (
    COMPACTED_TOOL_RESULT_MARKER,
    CONTEXT_COMPACT_NAMESPACE,
    SUMMARY_SYSTEM_PROMPT,
    ContextCompactConfig,
    ContextCompactor,
    is_prompt_too_long_error,
)
from harness.error_recovery import PromptTooLongRecoveryError
from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.model import ModelProvider, ModelRequest
from harness.state import AgentState
from services.artifacts import ArtifactStore
from services.checkpoint import create_in_memory_checkpointer


def system_prompt() -> str:
    """返回 Context Compact Graph 测试 Prompt。"""

    return "You are a context-compaction test assistant."


def chat_history(count: int) -> list[Message]:
    """创建没有重复项的交替对话历史。"""

    return [
        Message(
            role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
            content=f"message-{index}",
        )
        for index in range(count)
    ]


def test_large_tool_result_is_persisted_as_readable_artifact(tmp_path: Path) -> None:
    """大型 ToolResult 应写入 Artifact，并在消息中保留定位信息和预览。"""

    artifacts = ArtifactStore(tmp_path / "artifacts")
    model = FakeModel(Message(role=MessageRole.ASSISTANT, content="unused"))
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        artifacts,
        ContextCompactConfig(
            max_tool_result_characters=1_000,
            tool_result_preview_characters=100,
        ),
    )
    large_content = "source-data-" * 100
    state: AgentState = {
        "thread_id": "large-result-thread",
        "messages": [
            Message(role=MessageRole.USER, content="read data"),
            Message(
                role=MessageRole.ASSISTANT,
                tool_uses=(ToolUse(id="read-1", name="file_reader"),),
            ),
            Message(
                role=MessageRole.TOOL,
                tool_results=(ToolResult(tool_use_id="read-1", content=large_content),),
            ),
        ],
    }

    result = compactor.compact(state)

    compacted_content = result.messages[-1].tool_results[0].content
    assert isinstance(compacted_content, dict)
    assert compacted_content["compacted"] is True
    assert compacted_content["original_characters"] == len(large_content)
    relative_path = cast(str, compacted_content["artifact_path"])
    assert artifacts.resolve(relative_path).read_text(encoding="utf-8") == large_content
    assert model.sync_requests == []


def test_micro_compact_preserves_the_latest_tool_result(tmp_path: Path) -> None:
    """Micro Compact 应清理旧结果，但保留最近的 ToolUse/ToolResult。"""

    messages: list[Message] = []
    for index in range(3):
        tool_id = f"tool-{index}"
        messages.extend(
            (
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_uses=(ToolUse(id=tool_id, name="file_reader"),),
                ),
                Message(
                    role=MessageRole.TOOL,
                    tool_results=(
                        ToolResult(tool_use_id=tool_id, content=f"result-{index}-" * 80),
                    ),
                ),
            )
        )
    model = FakeModel(Message(role=MessageRole.ASSISTANT, content="unused"))
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        ArtifactStore(tmp_path / "artifacts"),
        ContextCompactConfig(
            keep_recent_tool_results=1,
            max_tool_result_characters=10_000,
            micro_compact_min_characters=100,
        ),
    )
    state: AgentState = {"thread_id": "micro-thread", "messages": messages}

    result = compactor.compact(state)

    first_content = result.messages[1].tool_results[0].content
    last_content = result.messages[-1].tool_results[0].content
    assert isinstance(first_content, dict)
    assert first_content["message"] == COMPACTED_TOOL_RESULT_MARKER
    assert last_content == "result-2-" * 80
    assert model.sync_requests == []


def test_summary_preserves_recent_tool_pair_and_runtime_state(tmp_path: Path) -> None:
    """摘要应保留近期工具配对、当前目标、Todo 与权限历史。"""

    tool_use = ToolUse(id="latest-tool", name="file_reader", input={"path": "source.txt"})
    messages = [
        *chat_history(6),
        Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
        Message(
            role=MessageRole.TOOL,
            tool_results=(ToolResult(tool_use_id=tool_use.id, content="latest evidence"),),
        ),
        Message(role=MessageRole.ASSISTANT, content="intermediate answer"),
        Message(role=MessageRole.USER, content="finish the current report"),
    ]
    model = FakeModel(
        Message(
            role=MessageRole.ASSISTANT,
            content="Old findings and source references were preserved.",
        )
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        artifacts,
        ContextCompactConfig(max_messages=8, keep_recent_messages=4),
    )
    state: AgentState = {
        "thread_id": "summary-thread",
        "messages": messages,
        "capability_state": {
            "todo_write": {"items": [{"content": "write report", "status": "pending"}]},
            "permission_history": [{"tool_name": "file_reader", "allowed": True}],
        },
    }

    result = compactor.compact(state)

    assert len(result.messages) == 5
    summary_message = result.messages[0]
    assert summary_message.content.startswith("[Context compacted: proactive]")
    assert "finish the current report" in summary_message.content
    assert "write report" in summary_message.content
    assert "permission_history" in summary_message.content
    assert result.messages[1].tool_uses == (tool_use,)
    assert result.messages[2].tool_results[0].tool_use_id == tool_use.id
    assert result.state.last_transcript_path is not None
    transcript = artifacts.resolve(result.state.last_transcript_path).read_text(
        encoding="utf-8"
    )
    assert len(transcript.splitlines()) == len(messages)
    assert len(model.sync_requests) == 1
    assert model.sync_requests[0].system_prompt == SUMMARY_SYSTEM_PROMPT


def test_summary_reads_tool_evidence_before_micro_compaction(tmp_path: Path) -> None:
    """摘要请求应看到旧工具证据，而不是只看到 Micro Compact 占位符。"""

    messages: list[Message] = []
    for index in range(4):
        tool_id = f"evidence-{index}"
        messages.extend(
            (
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_uses=(ToolUse(id=tool_id, name="file_reader"),),
                ),
                Message(
                    role=MessageRole.TOOL,
                    tool_results=(
                        ToolResult(
                            tool_use_id=tool_id,
                            content=f"important-evidence-{index}-" * 30,
                        ),
                    ),
                ),
            )
        )
    messages.extend(chat_history(2))
    model = FakeModel(Message(role=MessageRole.ASSISTANT, content="summary"))
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        ArtifactStore(tmp_path / "artifacts"),
        ContextCompactConfig(
            max_messages=8,
            keep_recent_messages=4,
            keep_recent_tool_results=1,
            max_tool_result_characters=10_000,
            micro_compact_min_characters=100,
        ),
    )
    state: AgentState = {"thread_id": "evidence-thread", "messages": messages}

    compactor.compact(state)

    summary_source = model.sync_requests[0].messages[0].content
    assert "important-evidence-0" in summary_source
    assert COMPACTED_TOOL_RESULT_MARKER not in summary_source


def test_graph_replaces_old_history_after_proactive_compaction(tmp_path: Path) -> None:
    """Graph 应使用 Overwrite 替换旧历史，不能让消息 Reducer 再次追加旧消息。"""

    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, content="compact summary"),
            Message(role=MessageRole.ASSISTANT, content="final answer"),
        ]
    )
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        ArtifactStore(tmp_path / "artifacts"),
        ContextCompactConfig(max_messages=8, keep_recent_messages=4),
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        context_compactor=compactor,
    )
    state: AgentState = {
        "thread_id": "proactive-graph-thread",
        "messages": chat_history(10),
    }

    result = loop.invoke(state)

    assert len(model.sync_requests) == 2
    assert len(result["messages"]) == 6
    assert result["messages"][0].content.startswith("[Context compacted: proactive]")
    assert all(message.content != "message-0" for message in result["messages"])
    assert result["messages"][-1].content == "final answer"
    capability_state = result.get("capability_state")
    assert capability_state is not None
    compact_state = capability_state[CONTEXT_COMPACT_NAMESPACE]
    assert isinstance(compact_state, dict)
    assert compact_state["compaction_count"] == 1


class PromptLengthRecoveryModel:
    """首次主请求超长、摘要和重试成功的测试模型。"""

    name = "prompt_length_recovery"

    def __init__(self, retry_succeeds: bool = True) -> None:
        self.retry_succeeds = retry_succeeds
        self.main_calls = 0
        self.summary_calls = 0

    def invoke(self, request: ModelRequest) -> Message:
        if request.system_prompt == SUMMARY_SYSTEM_PROMPT:
            self.summary_calls += 1
            return Message(role=MessageRole.ASSISTANT, content="reactive summary")
        self.main_calls += 1
        if self.main_calls == 1 or not self.retry_succeeds:
            raise RuntimeError("maximum context length exceeded")
        return Message(role=MessageRole.ASSISTANT, content="recovered answer")

    async def ainvoke(self, request: ModelRequest) -> Message:
        return self.invoke(request)


def test_prompt_too_long_compacts_and_retries_once(tmp_path: Path) -> None:
    """Prompt 过长时应压缩并仅补做一次主模型请求。"""

    model = PromptLengthRecoveryModel()
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        ArtifactStore(tmp_path / "artifacts"),
        ContextCompactConfig(
            max_context_characters=1_000_000,
            max_messages=120,
            keep_recent_messages=4,
        ),
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        context_compactor=compactor,
    )
    state: AgentState = {
        "thread_id": "reactive-thread",
        "messages": chat_history(10),
    }

    result = loop.invoke(state)

    assert model.main_calls == 2
    assert model.summary_calls == 1
    assert result["messages"][0].content.startswith("[Context compacted: reactive]")
    assert result["messages"][-1].content == "recovered answer"
    capability_state = result.get("capability_state")
    assert capability_state is not None
    compact_state = capability_state[CONTEXT_COMPACT_NAMESPACE]
    assert isinstance(compact_state, dict)
    assert compact_state["reactive_retry_count"] == 1


def test_prompt_too_long_retry_failure_is_not_retried_again(tmp_path: Path) -> None:
    """应急重试仍然超长时应直接返回错误，不能形成无限循环。"""

    model = PromptLengthRecoveryModel(retry_succeeds=False)
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        ArtifactStore(tmp_path / "artifacts"),
        ContextCompactConfig(
            max_context_characters=1_000_000,
            max_messages=120,
            keep_recent_messages=4,
        ),
    )
    loop = create_agent_loop(
        cast(ModelProvider, model),
        system_prompt,
        create_in_memory_checkpointer(),
        context_compactor=compactor,
    )
    state: AgentState = {
        "thread_id": "reactive-failure-thread",
        "messages": chat_history(10),
    }

    with pytest.raises(
        PromptTooLongRecoveryError,
        match="after reactive compact",
    ):
        loop.invoke(state)

    assert model.main_calls == 2
    assert model.summary_calls == 1


def test_prompt_too_long_error_detection_checks_wrapped_errors() -> None:
    """错误识别应支持常见标记和异常链。"""

    wrapped = RuntimeError("provider request failed")
    wrapped.__cause__ = ValueError("context_length_exceeded")

    assert is_prompt_too_long_error(wrapped)
    assert not is_prompt_too_long_error(RuntimeError("connection reset"))


def test_transcript_artifact_is_valid_json_lines(tmp_path: Path) -> None:
    """保存的完整 Transcript 应为可逐行恢复的 JSONL。"""

    artifacts = ArtifactStore(tmp_path / "artifacts")
    model = FakeModel(Message(role=MessageRole.ASSISTANT, content="summary"))
    compactor = ContextCompactor(
        cast(ModelProvider, model),
        artifacts,
        ContextCompactConfig(max_messages=8, keep_recent_messages=4),
    )
    state: AgentState = {
        "thread_id": "jsonl-thread",
        "messages": chat_history(10),
    }

    result = compactor.compact(state)

    assert result.state.last_transcript_path is not None
    lines = artifacts.resolve(result.state.last_transcript_path).read_text(
        encoding="utf-8"
    ).splitlines()
    restored = [json.loads(line) for line in lines]
    assert [item["content"] for item in restored] == [
        message.content for message in state["messages"]
    ]
