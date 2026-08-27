"""Knowledge Assistant 单 Agent 业务闭环集成测试。

Knowledge Assistant single-agent business-loop integration test.
"""

import logging
from pathlib import Path
from typing import Any, cast

import httpx
from langchain_core.runnables import RunnableConfig

from entrypoints.api import create_app
from entrypoints.bootstrap import create_agent_application
from harness.messages import Message, MessageRole, ToolUse
from harness.model import ModelProvider, ModelRequest
from harness.state import AgentState
from services.config import AppSettings, MemorySettings, RuntimePathSettings
from services.stores import SQLiteTaskStore


class ClosedLoopModel:
    """根据 ToolResult 推进完整文件分析闭环的确定性测试模型。"""

    name = "closed_loop_fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.task_id: str | None = None

    def invoke(self, request: ModelRequest) -> Message:
        raise AssertionError("closed-loop API test must use async model invocation")

    async def ainvoke(self, request: ModelRequest) -> Message:
        step = len(self.requests)
        self.requests.append(request)

        if step == 0:
            return self._tools(
                ToolUse(
                    id="todo-plan",
                    name="todo_write",
                    input={
                        "todos": [
                            {"content": "读取并比较本地资料", "status": "in_progress"},
                            {"content": "计算条目总数", "status": "pending"},
                            {"content": "保存报告并完成任务", "status": "pending"},
                        ]
                    },
                )
            )
        if step == 1:
            return self._tools(
                ToolUse(
                    id="task-create",
                    name="create_task",
                    input={
                        "title": "比较本地产品资料",
                        "description": "读取两个文件、计算条目数量并保存报告。",
                        "dependencies": [],
                    },
                )
            )
        if step == 2:
            self.task_id = self._latest_result(request)["task_id"]
            return self._tools(
                ToolUse(
                    id="task-claim",
                    name="claim_task",
                    input={"task_id": self.task_id, "owner": "knowledge_assistant"},
                )
            )
        if step == 3:
            return self._tools(
                ToolUse(
                    id="read-product",
                    name="file_reader",
                    input={"path": "product.txt"},
                ),
                ToolUse(
                    id="read-requirements",
                    name="file_reader",
                    input={"path": "requirements.txt"},
                ),
            )
        if step == 4:
            return self._tools(
                ToolUse(
                    id="count-items",
                    name="calculator",
                    input={"expression": "3 + 2"},
                )
            )
        if step == 5:
            return self._tools(
                ToolUse(
                    id="write-report",
                    name="report_writer",
                    input={
                        "path": "closed-loop-report.md",
                        "content": (
                            "# 本地资料比较报告\n\n"
                            "- 产品条目：3\n"
                            "- 需求条目：2\n"
                            "- 条目总数：5\n"
                            "- 结论：产品说明覆盖核心能力，需求清单补充验收要求。\n"
                        ),
                        "overwrite": True,
                    },
                )
            )
        if step == 6:
            if self.task_id is None:
                raise AssertionError("task ID must be available before completion")
            return self._tools(
                ToolUse(
                    id="task-complete",
                    name="complete_task",
                    input={
                        "task_id": self.task_id,
                        "owner": "knowledge_assistant",
                        "result_reference": "closed-loop-report.md",
                    },
                ),
                ToolUse(
                    id="follow-up-create",
                    name="create_task",
                    input={
                        "title": "复核本地资料比较报告",
                        "description": "后续人工复核报告结论。",
                        "dependencies": [self.task_id],
                    },
                ),
                ToolUse(
                    id="todo-complete",
                    name="todo_write",
                    input={
                        "todos": [
                            {"content": "读取并比较本地资料", "status": "completed"},
                            {"content": "计算条目总数", "status": "completed"},
                            {"content": "保存报告并完成任务", "status": "completed"},
                        ]
                    },
                ),
            )
        if step == 7:
            return Message(
                role=MessageRole.ASSISTANT,
                content="报告已保存为 closed-loop-report.md，分析任务已完成，后续复核任务已创建。",
            )
        raise AssertionError("closed-loop fake model response sequence exhausted")

    @staticmethod
    def _tools(*tool_uses: ToolUse) -> Message:
        return Message(role=MessageRole.ASSISTANT, tool_uses=tool_uses)

    @staticmethod
    def _latest_result(request: ModelRequest) -> dict[str, Any]:
        result = request.messages[-1].tool_results[0].content
        if not isinstance(result, dict):
            raise AssertionError("expected a structured task ToolResult")
        return cast(dict[str, Any], result)


async def test_knowledge_assistant_completes_single_agent_file_report_loop(
    tmp_path: Path,
    caplog,
) -> None:
    """单 Agent 应规划、执行、审批、保存报告并完成持久化 Task。"""

    workspace = tmp_path / "workspace"
    knowledge = workspace / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "product.txt").write_text(
        "产品条目：检索、比较、报告。",
        encoding="utf-8",
    )
    (knowledge / "requirements.txt").write_text(
        "需求条目：可追溯、需审批。",
        encoding="utf-8",
    )
    settings = AppSettings(
        environment="test",
        paths=RuntimePathSettings(
            workspace_root=workspace,
            knowledge_root=Path("knowledge"),
            artifact_root=Path("artifacts"),
        ),
        memory=MemorySettings(enabled=False),
        _env_file=None,
    )
    model = ClosedLoopModel()
    application = create_agent_application(
        model=cast(ModelProvider, model),
        settings=settings,
    )
    transport = httpx.ASGITransport(app=create_app(application))

    with caplog.at_level(logging.INFO, logger="harness.hooks"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/users/alice/conversations")
            conversation_id = created.json()["conversation_id"]
            runtime = application.runtimes.get("alice")
            report_path = runtime.artifact_root / "closed-loop-report.md"
            report_path.write_text("old report", encoding="utf-8")

            paused = await client.post(
                f"/users/alice/conversations/{conversation_id}/messages",
                json={
                    "content": (
                        "阅读 product.txt 和 requirements.txt，比较主要差异，计算条目总数，"
                        "覆盖保存 closed-loop-report.md，并记录一项后续复核任务。"
                    )
                },
            )

            assert paused.status_code == 200
            paused_body = paused.json()
            assert paused_body["status"] == "waiting_permission"
            assert paused_body["permission_request"]["requests"][0]["tool_name"] == (
                "report_writer"
            )

            resumed = await client.post(
                f"/users/alice/conversations/{conversation_id}/runs/"
                f"{paused_body['run_id']}/permission",
                json={"approved": True},
            )

    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "idle"
    assert body["messages"][-1]["content"].startswith("报告已保存")
    assert report_path.read_text(encoding="utf-8").startswith("# 本地资料比较报告")

    tool_uses = [
        use
        for message in body["messages"]
        for use in message.get("tool_uses", [])
    ]
    tool_results = [
        result
        for message in body["messages"]
        for result in message.get("tool_results", [])
    ]
    assert [use["name"] for use in tool_uses] == [
        "todo_write",
        "create_task",
        "claim_task",
        "file_reader",
        "file_reader",
        "calculator",
        "report_writer",
        "complete_task",
        "create_task",
        "todo_write",
    ]
    assert {use["id"] for use in tool_uses} == {
        result["tool_use_id"] for result in tool_results
    }
    assert not any(result["is_error"] for result in tool_results)

    task_store = SQLiteTaskStore(runtime.task_database_path)
    try:
        tasks = task_store.list()
    finally:
        task_store.close()
    completed = next(task for task in tasks if task.title == "比较本地产品资料")
    follow_up = next(task for task in tasks if task.title == "复核本地资料比较报告")
    assert completed.status.value == "completed"
    assert completed.result_reference == "closed-loop-report.md"
    assert completed.conversation_id == conversation_id
    assert completed.run_id == paused_body["run_id"]
    assert follow_up.status.value == "pending"
    assert follow_up.dependencies == (completed.task_id,)
    assert follow_up.conversation_id == conversation_id
    assert follow_up.run_id == paused_body["run_id"]

    snapshot_config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
    snapshot = await runtime.agent_loop.graph.aget_state(snapshot_config)
    state = cast(AgentState, snapshot.values)
    permission_history = state["capability_state"]["permission_history"]
    assert isinstance(permission_history, list)
    report_permission = next(
        item
        for item in permission_history
        if isinstance(item, dict) and item.get("tool_use_id") == "write-report"
    )
    assert report_permission["decision"] == "ask"
    assert report_permission["allowed"] is True

    tool_logs = [record for record in caplog.records if record.message.startswith("tool call")]
    assert tool_logs
    assert {record.thread_id for record in tool_logs} == {conversation_id}
    assert {record.run_id for record in tool_logs} == {paused_body["run_id"]}

    available_tools = {tool.name for tool in model.requests[0].tools}
    assert "document_search" not in available_tools
    assert not any("subagent" in name or "team" in name for name in available_tools)
    assert "complete one\nclosed loop" in model.requests[0].system_prompt
    assert len(model.requests) == 8
