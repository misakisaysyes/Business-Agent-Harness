"""ConversationService 和 Agent Server 集成测试。

ConversationService and agent-server integration tests.
"""

import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from tests.fakes import FakeSequenceModel

from entrypoints.api import _raise_http_error, create_app
from entrypoints.bootstrap import create_agent_application
from harness.capabilities.memory import MemoryDraft, MemorySearchTool, MemoryType
from harness.error_recovery import OutputTokenRecoveryError, PromptTooLongRecoveryError
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider, ModelRequest
from services.config import AppSettings, RuntimePathSettings
from services.model_gateway import (
    ModelGatewayEvent,
    ModelGatewayEventType,
    ModelGatewayUnavailableError,
)
from services.stores import FileMemoryStore, SQLiteTaskStore


class EchoHistoryModel:
    """返回最后一条消息及当前 Thread 用户消息数量。"""

    name = "echo_history"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("API tests should use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        self.requests.append(request)
        user_messages = [
            message for message in request.messages if message.role is MessageRole.USER
        ]
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"{user_messages[-1].content}:{len(user_messages)}",
        )


class BlockingModel:
    """在测试释放前保持一个模型请求运行。"""

    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("API tests should use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        self.started.set()
        await self.release.wait()
        return Message(role=MessageRole.ASSISTANT, content="done")


class ConcurrentModel:
    """等待两个请求同时进入，以证明不同 Conversation 可并发。"""

    name = "concurrent"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.both_started = asyncio.Event()

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("API tests should use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.both_started.set()
        await asyncio.wait_for(self.both_started.wait(), timeout=2)
        self.active -= 1
        return Message(role=MessageRole.ASSISTANT, content="done")


class UsageEchoModel:
    """返回固定真实用量元数据的测试模型。"""

    name = "usage_echo"

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("API tests should use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        return Message(
            role=MessageRole.ASSISTANT,
            content="done",
            provider_metadata={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                }
            },
        )


def settings(tmp_path: Path) -> AppSettings:
    workspace = tmp_path / "workspace"
    (workspace / "knowledge").mkdir(parents=True)
    return AppSettings(
        environment="test",
        paths=RuntimePathSettings(
            workspace_root=workspace,
            knowledge_root=Path("knowledge"),
            artifact_root=Path("artifacts"),
        ),
        _env_file=None,
    )


async def test_users_have_isolated_runtime_history_and_artifact_roots(tmp_path: Path) -> None:
    """Alice/Bob 应共享模型，但隔离 Runtime、历史和 Artifact。"""

    model = EchoHistoryModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        bob_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]

        alice_first = await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "alice-one"},
        )
        bob_first = await client.post(
            f"/users/bob/conversations/{bob_id}/messages",
            json={"content": "bob-one"},
        )
        alice_second = await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "alice-two"},
        )
        forbidden = await client.post(
            f"/users/bob/conversations/{alice_id}/messages",
            json={"content": "steal"},
        )
        invalid_user = await client.post("/users/alice.invalid/conversations")

    assert alice_first.json()["messages"][-1]["content"] == "alice-one:1"
    assert bob_first.json()["messages"][-1]["content"] == "bob-one:1"
    assert alice_second.json()["messages"][-1]["content"] == "alice-two:2"
    assert forbidden.status_code == 403
    assert invalid_user.status_code == 422

    alice_runtime = application.runtimes.get("alice")
    bob_runtime = application.runtimes.get("bob")
    assert application.runtimes.get("alice") is alice_runtime
    assert alice_runtime.agent_loop is not bob_runtime.agent_loop
    assert alice_runtime.artifact_root == tmp_path / "workspace/artifacts/alice"
    assert bob_runtime.artifact_root == tmp_path / "workspace/artifacts/bob"
    assert alice_runtime.workspace_root == tmp_path / "workspace/users/alice/workspace"
    assert bob_runtime.workspace_root == tmp_path / "workspace/users/bob/workspace"
    assert alice_runtime.private_knowledge_root != bob_runtime.private_knowledge_root
    assert alice_runtime.private_skills_root != bob_runtime.private_skills_root
    assert alice_runtime.memory_root != bob_runtime.memory_root
    assert alice_runtime.task_database_path != bob_runtime.task_database_path
    assert all(
        path.is_dir()
        for path in (
            alice_runtime.workspace_root,
            alice_runtime.private_knowledge_root,
            alice_runtime.private_skills_root,
            alice_runtime.memory_root,
            alice_runtime.artifact_root,
        )
    )

    alice_tasks = SQLiteTaskStore(alice_runtime.task_database_path)
    bob_tasks = SQLiteTaskStore(bob_runtime.task_database_path)
    try:
        alice_tasks.create("Alice private task")
        assert len(alice_tasks.list()) == 1
        assert bob_tasks.list() == ()
    finally:
        alice_tasks.close()
        bob_tasks.close()


async def test_api_can_force_an_available_tool_for_the_first_model_call(
    tmp_path: Path,
) -> None:
    """API required_tool 应校验名称，并只传入当前 Turn 的 ModelRequest。"""

    model = EchoHistoryModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        forced = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "calculate", "required_tool": "calculator"},
        )
        unknown = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "invalid", "required_tool": "missing_tool"},
        )

    assert forced.status_code == 200
    assert model.requests[0].required_tool == "calculator"
    assert unknown.status_code == 422
    assert "required tool is not available" in unknown.json()["detail"]
    assert len(model.requests) == 1


async def test_api_reports_disabled_mcp_without_exposing_configuration(
    tmp_path: Path,
) -> None:
    """默认关闭 MCP 时应返回稳定空清单。"""

    application = create_agent_application(
        model=cast(ModelProvider, EchoHistoryModel()),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/users/alice/mcp")

    assert response.status_code == 200
    assert response.json() == {"enabled": False, "servers": [], "tools": []}


async def test_cross_session_memory_is_shared_per_user_but_isolated_between_users(
    tmp_path: Path,
) -> None:
    """Alice 的 Memory 应跨会话可用，但不得进入 Bob 的 Prompt。"""

    model = EchoHistoryModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    alice_runtime = application.runtimes.get("alice")
    bob_runtime = application.runtimes.get("bob")
    FileMemoryStore(alice_runtime.memory_root).upsert(
        MemoryDraft(
            name="response-style",
            memory_type=MemoryType.USER,
            description="Alice 的回答偏好",
            content="ALICE-MEMORY-ONLY Alice 偏好中文简短回答。",
            tags=("回答", "偏好"),
            source="alice conversation",
        ),
        "alice-memory-write",
    )

    await alice_runtime.agent_loop.ainvoke(
        {
            "thread_id": "alice-new-conversation",
            "messages": [Message(role=MessageRole.USER, content="我的回答偏好是什么？")],
        }
    )
    await bob_runtime.agent_loop.ainvoke(
        {
            "thread_id": "bob-new-conversation",
            "messages": [Message(role=MessageRole.USER, content="我的回答偏好是什么？")],
        }
    )

    alice_prompt = model.requests[-2].system_prompt
    bob_prompt = model.requests[-1].system_prompt
    assert "ALICE-MEMORY-ONLY" in alice_prompt
    assert "ALICE-MEMORY-ONLY" not in bob_prompt

    bob_search = await MemorySearchTool(FileMemoryStore(bob_runtime.memory_root)).ainvoke(
        ToolUse(
            id="bob-memory-search",
            name="memory_search",
            input={"query": "回答偏好"},
        )
    )
    assert "ALICE-MEMORY-ONLY" not in str(bob_search.content)


async def test_private_knowledge_and_skill_summaries_are_user_scoped(tmp_path: Path) -> None:
    """每个用户只能在 Prompt 中看到自己的私有 Knowledge 和 Skill。"""

    active_settings = settings(tmp_path)
    users_root = active_settings.paths.workspace_root / "users"
    alice_knowledge = users_root / "alice/knowledge"
    bob_knowledge = users_root / "bob/knowledge"
    alice_skill = users_root / "alice/skills/alice-workflow"
    bob_skill = users_root / "bob/skills/bob-workflow"
    for path in (alice_knowledge, bob_knowledge, alice_skill, bob_skill):
        path.mkdir(parents=True)
    (alice_knowledge / "alice-private.txt").write_text("alice secret", encoding="utf-8")
    (bob_knowledge / "bob-private.txt").write_text("bob secret", encoding="utf-8")
    (alice_skill / "SKILL.md").write_text(
        "---\nname: alice-workflow\ndescription: Alice private workflow.\n---\nAlice body.\n",
        encoding="utf-8",
    )
    (bob_skill / "SKILL.md").write_text(
        "---\nname: bob-workflow\ndescription: Bob private workflow.\n---\nBob body.\n",
        encoding="utf-8",
    )
    model = EchoHistoryModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=active_settings,
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        bob_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]
        await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "list resources"},
        )
        await client.post(
            f"/users/bob/conversations/{bob_id}/messages",
            json={"content": "list resources"},
        )
        alice_skills = await client.get("/users/alice/skills")
        bob_skills = await client.get("/users/bob/skills")

    alice_prompt = model.requests[0].system_prompt
    bob_prompt = model.requests[1].system_prompt
    assert "alice-private.txt" in alice_prompt
    assert "alice-workflow" in alice_prompt
    assert "bob-private.txt" not in alice_prompt
    assert "bob-workflow" not in alice_prompt
    assert "bob-private.txt" in bob_prompt
    assert "bob-workflow" in bob_prompt
    assert "alice-private.txt" not in bob_prompt
    assert "alice-workflow" not in bob_prompt
    assert {item["name"] for item in alice_skills.json()["skills"]} >= {
        "knowledge-synthesis",
        "alice-workflow",
    }
    assert "bob-workflow" not in {
        item["name"] for item in alice_skills.json()["skills"]
    }
    assert "bob-workflow" in {item["name"] for item in bob_skills.json()["skills"]}
    assert "alice-workflow" not in {
        item["name"] for item in bob_skills.json()["skills"]
    }


async def test_file_reader_allows_owner_private_knowledge_and_denies_other_user(
    tmp_path: Path,
) -> None:
    """Alice 的私有资料可自动读取，Bob 读取相同绝对路径必须被拒绝。"""

    active_settings = settings(tmp_path)
    alice_file = active_settings.paths.workspace_root / "users/alice/knowledge/private.txt"
    alice_file.parent.mkdir(parents=True)
    alice_file.write_text("ALICE-PRIVATE-CONTENT", encoding="utf-8")
    alice_call = ToolUse(
        id="alice-read",
        name="file_reader",
        input={"path": str(alice_file)},
    )
    bob_call = ToolUse(
        id="bob-read",
        name="file_reader",
        input={"path": str(alice_file)},
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(alice_call,)),
            Message(role=MessageRole.ASSISTANT, content="alice done"),
            Message(role=MessageRole.ASSISTANT, tool_uses=(bob_call,)),
            Message(role=MessageRole.ASSISTANT, content="bob denied"),
        ]
    )
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=active_settings,
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        bob_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]
        alice_response = await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "read my private file"},
        )
        bob_response = await client.post(
            f"/users/bob/conversations/{bob_id}/messages",
            json={"content": "read Alice private file"},
        )

    alice_results = alice_response.json()["messages"][-2]["tool_results"]
    bob_results = bob_response.json()["messages"][-2]["tool_results"]
    assert alice_results[0]["content"] == "ALICE-PRIVATE-CONTENT"
    assert bob_results[0]["is_error"] is True
    assert bob_results[0]["content"]["error"] == "permission_denied"
    assert "ALICE-PRIVATE-CONTENT" not in str(bob_results)


async def test_token_usage_is_accounted_per_user(tmp_path: Path) -> None:
    """共享模型返回的真实 Token 用量必须分别累计到 Alice 和 Bob。"""

    application = create_agent_application(
        model=cast(ModelProvider, UsageEchoModel()),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        alice_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        bob_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]
        alice_first = await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "one"},
        )
        alice_second = await client.post(
            f"/users/alice/conversations/{alice_id}/messages",
            json={"content": "two"},
        )
        await client.post(
            f"/users/bob/conversations/{bob_id}/messages",
            json={"content": "one"},
        )
        alice_total = await client.get("/users/alice/usage")
        bob_total = await client.get("/users/bob/usage")

    assert alice_first.json()["token_usage"]["total_tokens"] == 12
    assert alice_second.json()["token_usage"]["total_tokens"] == 12
    assert alice_total.json()["token_usage"] == {
        "input_tokens": 20,
        "output_tokens": 4,
        "total_tokens": 24,
    }
    assert bob_total.json()["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }


async def test_list_and_delete_conversations_are_owner_scoped(tmp_path: Path) -> None:
    """列表只返回当前用户数据，删除应清理 Registry 和 Checkpoint。"""

    application = create_agent_application(
        model=cast(ModelProvider, EchoHistoryModel()),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        second_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        bob_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]
        await client.post(
            f"/users/alice/conversations/{first_id}/messages",
            json={"content": "remember this"},
        )

        checkpoint_config: RunnableConfig = {"configurable": {"thread_id": first_id}}
        checkpointer = application.runtimes.get("alice").agent_loop.checkpointer
        assert await checkpointer.aget_tuple(checkpoint_config) is not None

        listed = await client.get("/users/alice/conversations")
        forbidden = await client.delete(f"/users/bob/conversations/{first_id}")
        deleted = await client.delete(f"/users/alice/conversations/{first_id}")
        listed_after_delete = await client.get("/users/alice/conversations")
        missing = await client.post(
            f"/users/alice/conversations/{first_id}/messages",
            json={"content": "should fail"},
        )

    listed_ids = {item["conversation_id"] for item in listed.json()["conversations"]}
    remaining_ids = {
        item["conversation_id"] for item in listed_after_delete.json()["conversations"]
    }
    assert listed_ids == {first_id, second_id}
    assert bob_id not in listed_ids
    assert forbidden.status_code == 403
    assert deleted.status_code == 204
    assert remaining_ids == {second_id}
    assert missing.status_code == 404
    assert await checkpointer.aget_tuple(checkpoint_config) is None


async def test_api_does_not_expose_internal_provider_metadata(tmp_path: Path) -> None:
    """模型回传所需的内部元数据不应暴露给 CLI。"""

    model = FakeSequenceModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content="answer",
                provider_metadata={"reasoning_content": "private thinking state"},
            )
        ]
    )
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        response = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "hello"},
        )

    assert response.status_code == 200
    assert "provider_metadata" not in response.json()["messages"][-1]


def test_gateway_failure_detail_contains_visible_model_events() -> None:
    """最终 503 应携带 CLI 可展示的模型重试事件。"""

    event = ModelGatewayEvent(
        event_type=ModelGatewayEventType.RETRY,
        model="moonshot/kimi-k3",
        reason="TimeoutError",
        retry_number=2,
        max_retries=2,
        delay_seconds=2.0,
    )

    with pytest.raises(HTTPException) as captured:
        _raise_http_error(
            ModelGatewayUnavailableError("all model routes failed"),
            [event],
        )

    assert captured.value.status_code == 503
    assert captured.value.detail["message"] == "all model routes failed"
    assert captured.value.detail["model_events"][0]["retry_number"] == 2


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (PromptTooLongRecoveryError("prompt too long after compact"), 413),
        (OutputTokenRecoveryError("output still truncated"), 502),
    ],
)
def test_bounded_recovery_failures_have_stable_http_statuses(
    error: Exception,
    expected_status: int,
) -> None:
    """恢复耗尽应返回明确状态码，而不是泄漏内部 500。"""

    with pytest.raises(HTTPException) as captured:
        _raise_http_error(error)

    assert captured.value.status_code == expected_status
    assert captured.value.detail == str(error)


async def test_same_conversation_rejects_a_second_active_run(tmp_path: Path) -> None:
    """同一 Conversation 的第二个活跃请求应立即返回 409。"""

    model = BlockingModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        first_request = asyncio.create_task(
            client.post(
                f"/users/alice/conversations/{conversation_id}/messages",
                json={"content": "first"},
            )
        )
        await asyncio.wait_for(model.started.wait(), timeout=2)
        conflict = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "second"},
        )
        model.release.set()
        completed = await first_request

    assert conflict.status_code == 409
    assert completed.status_code == 200


async def test_running_conversation_can_be_cancelled_and_reused(tmp_path: Path) -> None:
    """用户取消应终止活跃 Graph Task，并把 Conversation 恢复为空闲。"""

    model = BlockingModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        active_request = asyncio.create_task(
            client.post(
                f"/users/alice/conversations/{conversation_id}/messages",
                json={"content": "long request"},
            )
        )
        await asyncio.wait_for(model.started.wait(), timeout=2)
        detail = await client.get(f"/users/alice/conversations/{conversation_id}")
        run_id = detail.json()["active_run_id"]

        cancelled = await client.post(
            f"/users/alice/conversations/{conversation_id}/runs/{run_id}/cancel"
        )
        original_response = await active_request
        model.release.set()
        reused = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "new request"},
        )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "idle"
    assert cancelled.json()["active_run_id"] is None
    assert original_response.status_code == 409
    assert "cancelled" in original_response.json()["detail"]
    assert reused.status_code == 200


async def test_different_conversations_execute_concurrently(tmp_path: Path) -> None:
    """不同 Conversation 应进入两个并发 Agent Run。"""

    model = ConcurrentModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings(tmp_path),
    )
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first_id = (await client.post("/users/alice/conversations")).json()["conversation_id"]
        second_id = (await client.post("/users/bob/conversations")).json()["conversation_id"]
        responses = await asyncio.gather(
            client.post(
                f"/users/alice/conversations/{first_id}/messages",
                json={"content": "first"},
            ),
            client.post(
                f"/users/bob/conversations/{second_id}/messages",
                json={"content": "second"},
            ),
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert model.max_active == 2


async def test_permission_resume_keeps_conversation_and_run_identity(tmp_path: Path) -> None:
    """Permission 恢复必须沿用原 Conversation ID 和 Run ID。"""

    active_settings = settings(tmp_path)
    tool_use = ToolUse(
        id="read-private",
        name="file_reader",
        input={"path": "private.txt"},
    )
    model = FakeSequenceModel(
        [
            Message(role=MessageRole.ASSISTANT, tool_uses=(tool_use,)),
            Message(role=MessageRole.ASSISTANT, content="读取完成"),
        ]
    )
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=active_settings,
    )
    alice_runtime = application.runtimes.get("alice")
    (alice_runtime.workspace_root / "private.txt").write_text("approved", encoding="utf-8")
    transport = httpx.ASGITransport(app=create_app(application))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/users/alice/conversations")).json()[
            "conversation_id"
        ]
        paused = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "读取 private.txt"},
        )
        paused_body = paused.json()
        run_id = paused_body["run_id"]

        conflict = await client.post(
            f"/users/alice/conversations/{conversation_id}/messages",
            json={"content": "another message"},
        )
        delete_conflict = await client.delete(f"/users/alice/conversations/{conversation_id}")
        forbidden = await client.post(
            f"/users/bob/conversations/{conversation_id}/runs/{run_id}/permission",
            json={"approved": True},
        )
        resumed = await client.post(
            f"/users/alice/conversations/{conversation_id}/runs/{run_id}/permission",
            json={"approved": True},
        )

    assert paused_body["status"] == "waiting_permission"
    assert paused_body["permission_request"]["requests"][0]["tool_name"] == "file_reader"
    assert conflict.status_code == 409
    assert delete_conflict.status_code == 409
    assert forbidden.status_code == 403
    assert resumed.status_code == 200
    assert resumed.json()["run_id"] == run_id
    assert resumed.json()["status"] == "idle"
    assert resumed.json()["messages"][-1]["content"] == "读取完成"
