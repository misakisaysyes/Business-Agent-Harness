"""M4-5 SQLite Checkpoint 恢复集成测试。

M4-5 SQLite checkpoint recovery integration tests.
"""

from pathlib import Path
from typing import cast

import httpx
from pydantic import JsonValue
from tests.fakes import FakeSequenceModel

from entrypoints.api import create_app
from entrypoints.bootstrap import bootstrap_agent, create_agent_application
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider, ModelRequest
from services.checkpoint import create_sqlite_checkpointer
from services.config import AppSettings, RuntimePathSettings


class RestartHistoryModel:
    """返回 Checkpoint 中可见的用户消息数量。"""

    name = "restart_history"

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("API tests should use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        user_count = sum(
            message.role is MessageRole.USER for message in request.messages
        )
        return Message(role=MessageRole.ASSISTANT, content=f"users={user_count}")


def checkpoint_settings(tmp_path: Path) -> AppSettings:
    """创建每个测试独占的 Workspace 和 SQLite 数据库路径。"""

    workspace = tmp_path / "workspace"
    (workspace / "knowledge").mkdir(parents=True)
    return AppSettings(  # pyright: ignore[reportCallIssue]
        environment="test",
        paths=RuntimePathSettings(
            workspace_root=workspace,
            knowledge_root=Path("knowledge"),
            artifact_root=Path("artifacts"),
        ),
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )


async def test_conversation_history_survives_application_recreation(tmp_path: Path) -> None:
    """重建应用后，同一 Conversation 应从 SQLite 读取此前消息。"""

    active_settings = checkpoint_settings(tmp_path)
    first_application = create_agent_application(
        model=cast(ModelProvider, RestartHistoryModel()),
        settings=active_settings,
    )
    first_transport = httpx.ASGITransport(app=create_app(first_application))
    async with httpx.AsyncClient(transport=first_transport, base_url="http://first") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        first = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "first"},
        )
    assert first.json()["messages"][-1]["content"] == "users=1"

    # 创建全新的 Application/Runtime，模拟 Server 进程重启后的依赖重建。
    # Build a new application/runtime to simulate dependency recreation after restart.
    second_application = create_agent_application(
        model=cast(ModelProvider, RestartHistoryModel()),
        settings=active_settings,
    )
    second_transport = httpx.ASGITransport(app=create_app(second_application))
    async with httpx.AsyncClient(transport=second_transport, base_url="http://second") as client:
        listed = await client.get("/users/alice/conversations")
        second = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "second"},
        )

    assert listed.json()["conversations"] == [
        {"conversation_id": conversation_id, "status": "idle"}
    ]
    assert second.json()["messages"][-1]["content"] == "users=2"
    assert (active_settings.paths.workspace_root / "users/alice/checkpoints.sqlite3").is_file()
    assert (
        active_settings.paths.workspace_root / ".agent/conversations.sqlite3"
    ).is_file()


async def test_capability_state_round_trips_through_sqlite(tmp_path: Path) -> None:
    """Todo、Task 引用和 Compact 状态应随完整 AgentState 持久化。"""

    active_settings = checkpoint_settings(tmp_path)
    database_path = tmp_path / "state-round-trip.sqlite3"
    capability_state: dict[str, JsonValue] = {
        "todo_write": {"items": [{"content": "inspect", "status": "pending"}]},
        "task_refs": ["task-001"],
        "context_compact": {"summary": "earlier context"},
    }
    first_saver = create_sqlite_checkpointer(database_path)
    first_loop = bootstrap_agent(
        model=cast(
            ModelProvider,
            FakeSequenceModel([Message(role=MessageRole.ASSISTANT, content="first")]),
        ),
        settings=active_settings,
        checkpointer=first_saver,
    )
    await first_loop.ainvoke(
        {
            "thread_id": "stable-thread",
            "messages": [Message(role=MessageRole.USER, content="first")],
            "capability_state": capability_state,
        }
    )
    first_saver.close()

    second_saver = create_sqlite_checkpointer(database_path)
    second_loop = bootstrap_agent(
        model=cast(
            ModelProvider,
            FakeSequenceModel([Message(role=MessageRole.ASSISTANT, content="second")]),
        ),
        settings=active_settings,
        checkpointer=second_saver,
    )
    restored = await second_loop.ainvoke(
        {
            "thread_id": "stable-thread",
            "messages": [Message(role=MessageRole.USER, content="second")],
        }
    )
    second_saver.close()

    assert restored.get("capability_state") == capability_state
    restored_user_messages = [
        message.content
        for message in restored["messages"]
        if message.role is MessageRole.USER
    ]
    assert restored_user_messages == [
        "first",
        "second",
    ]


async def test_permission_interrupt_resumes_after_application_recreation(
    tmp_path: Path,
) -> None:
    """Permission interrupt 应以原 thread_id/run_id 跨重建恢复。"""

    active_settings = checkpoint_settings(tmp_path)
    overwrite = ToolUse(
        id="checkpoint-write",
        name="report_writer",
        input={
            "path": "checkpoint-report.md",
            "content": "new content",
            "overwrite": True,
        },
    )
    first_model = FakeSequenceModel(
        [Message(role=MessageRole.ASSISTANT, tool_uses=(overwrite,))]
    )
    first_application = create_agent_application(
        model=cast(ModelProvider, first_model),
        settings=active_settings,
    )
    artifact = first_application.runtimes.get("alice").artifact_root / "checkpoint-report.md"
    artifact.write_text("old content", encoding="utf-8")
    first_transport = httpx.ASGITransport(app=create_app(first_application))
    async with httpx.AsyncClient(transport=first_transport, base_url="http://first") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        paused = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "overwrite the report"},
        )
    paused_body = paused.json()
    run_id = paused_body["run_id"]
    assert paused_body["status"] == "waiting_permission"
    assert artifact.read_text(encoding="utf-8") == "old content"

    second_model = FakeSequenceModel(
        [Message(role=MessageRole.ASSISTANT, content="write completed")]
    )
    second_application = create_agent_application(
        model=cast(ModelProvider, second_model),
        settings=active_settings,
    )
    second_transport = httpx.ASGITransport(app=create_app(second_application))
    async with httpx.AsyncClient(transport=second_transport, base_url="http://second") as client:
        detail = await client.get(f"/users/alice/conversations/{conversation_id}")
        resumed = await client.post(
            f"/users/alice/conversations/{conversation_id}/runs/{run_id}/permission",
            json={"approved": True},
        )
        duplicate_resume = await client.post(
            f"/users/alice/conversations/{conversation_id}/runs/{run_id}/permission",
            json={"approved": True},
        )

    assert detail.json()["status"] == "waiting_permission"
    assert detail.json()["active_run_id"] == run_id
    assert detail.json()["permission_request"]["requests"][0]["tool_name"] == "report_writer"
    assert resumed.status_code == 200
    assert resumed.json()["conversation_id"] == conversation_id
    assert resumed.json()["run_id"] == run_id
    assert resumed.json()["status"] == "idle"
    assert artifact.read_text(encoding="utf-8") == "new content"
    assert duplicate_resume.status_code == 409
    assert len(second_model.async_requests) == 1
