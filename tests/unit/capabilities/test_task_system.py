"""Task System 生命周期、依赖、并发与持久化测试。

Task System lifecycle, dependency, concurrency, and persistence tests.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from harness.capabilities.task_system import (
    InMemoryTaskStore,
    TaskDependencyError,
    TaskStatus,
    TaskTransitionError,
    create_task_tools,
)
from harness.messages import ToolUse
from services.stores import SQLiteTaskStore


def test_task_ids_are_unique_and_lifecycle_fields_are_persisted() -> None:
    """Task ID 应唯一，合法生命周期应保留 Owner 和结果引用。"""

    store = InMemoryTaskStore()
    first = store.create("读取资料", "读取授权文件")
    second = store.create("生成报告")

    assert first.task_id != second.task_id
    assert first.status is TaskStatus.PENDING

    claimed = store.claim(first.task_id, "knowledge_assistant")
    completed = store.complete(
        first.task_id,
        "knowledge_assistant",
        "artifacts/report.md",
    )

    assert claimed.status is TaskStatus.IN_PROGRESS
    assert completed.status is TaskStatus.COMPLETED
    assert completed.owner == "knowledge_assistant"
    assert completed.result_reference == "artifacts/report.md"


def test_task_dependencies_block_claim_until_completed() -> None:
    """依赖 Task 完成之前，下游 Task 不得被认领。"""

    store = InMemoryTaskStore()
    prerequisite = store.create("读取资料")
    dependent = store.create("生成报告", dependencies=(prerequisite.task_id,))

    with pytest.raises(TaskDependencyError, match="not completed"):
        store.claim(dependent.task_id, "worker-b")

    store.claim(prerequisite.task_id, "worker-a")
    store.complete(prerequisite.task_id, "worker-a", "knowledge/source.txt")

    assert store.claim(dependent.task_id, "worker-b").owner == "worker-b"


def test_task_rejects_invalid_transitions_and_wrong_owner() -> None:
    """终态转换和完成操作必须遵守状态及 Owner 约束。"""

    store = InMemoryTaskStore()
    task = store.create("生成报告")

    with pytest.raises(TaskTransitionError, match="not in_progress"):
        store.complete(task.task_id, "worker-a", "report.md")

    store.claim(task.task_id, "worker-a")

    with pytest.raises(TaskTransitionError, match="owner mismatch"):
        store.fail(task.task_id, "worker-b", "failed")

    store.fail(task.task_id, "worker-a", "source was unavailable")

    with pytest.raises(TaskTransitionError, match="not pending"):
        store.claim(task.task_id, "worker-c")


async def test_generic_task_tools_expose_the_complete_workflow() -> None:
    """通用工具应直接覆盖创建、读取、列表、认领、完成和失败操作。"""

    store = InMemoryTaskStore()
    tools = {tool.name: tool for tool in create_task_tools(store)}

    assert tuple(tools) == (
        "create_task",
        "get_task",
        "list_tasks",
        "claim_task",
        "complete_task",
        "fail_task",
    )

    created_result = await tools["create_task"].ainvoke(
        ToolUse(
            id="create-1",
            name="create_task",
            input={"title": "生成报告", "description": "保存最终报告"},
        )
    )
    assert isinstance(created_result.content, dict)
    task_id = str(created_result.content["task_id"])

    await tools["claim_task"].ainvoke(
        ToolUse(
            id="claim-1",
            name="claim_task",
            input={"task_id": task_id, "owner": "knowledge_assistant"},
        )
    )
    completed_result = await tools["complete_task"].ainvoke(
        ToolUse(
            id="complete-1",
            name="complete_task",
            input={
                "task_id": task_id,
                "owner": "knowledge_assistant",
                "result_reference": "artifacts/report.md",
            },
        )
    )
    listed_result = await tools["list_tasks"].ainvoke(
        ToolUse(
            id="list-1",
            name="list_tasks",
            input={"status": "completed"},
        )
    )

    assert isinstance(completed_result.content, dict)
    assert completed_result.content["status"] == "completed"
    assert isinstance(listed_result.content, list)
    assert len(listed_result.content) == 1


def test_concurrent_sqlite_claim_allows_only_one_owner(tmp_path: Path) -> None:
    """两个独立 Store 同时认领时只能有一个成功。"""

    database_path = tmp_path / "tasks.sqlite3"
    first_store = SQLiteTaskStore(database_path)
    second_store = SQLiteTaskStore(database_path)
    task = first_store.create("并发认领测试")
    barrier = Barrier(2)

    def claim(store: SQLiteTaskStore, owner: str) -> str:
        barrier.wait()
        try:
            return store.claim(task.task_id, owner).owner or ""
        except TaskTransitionError:
            return "rejected"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(
                executor.map(
                    lambda item: claim(*item),
                    ((first_store, "worker-a"), (second_store, "worker-b")),
                )
            )

        assert outcomes.count("rejected") == 1
        assert set(outcomes) & {"worker-a", "worker-b"}
        assert first_store.get(task.task_id).owner in {"worker-a", "worker-b"}
    finally:
        first_store.close()
        second_store.close()


def test_sqlite_store_survives_restart_with_result_reference(tmp_path: Path) -> None:
    """关闭并重建 Store 后仍应保留 Task 状态和结果引用。"""

    database_path = tmp_path / "tasks.sqlite3"
    first_store = SQLiteTaskStore(database_path)
    task = first_store.create("持久化报告")
    first_store.claim(task.task_id, "knowledge_assistant")
    first_store.complete(task.task_id, "knowledge_assistant", "artifacts/final.md")
    first_store.close()

    restarted_store = SQLiteTaskStore(database_path)
    try:
        restored = restarted_store.get(task.task_id)

        assert restored.status is TaskStatus.COMPLETED
        assert restored.owner == "knowledge_assistant"
        assert restored.result_reference == "artifacts/final.md"
    finally:
        restarted_store.close()


def test_sqlite_store_migrates_legacy_tasks_and_persists_run_correlation(
    tmp_path: Path,
) -> None:
    """旧表应无损增加关联列，新 Task 应持久化 Conversation 和 Run 来源。"""

    database_path = tmp_path / "legacy-tasks.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            dependencies TEXT NOT NULL,
            owner TEXT,
            result_reference TEXT,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_id, title, description, status, dependencies, owner,
            result_reference, failure_reason, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task_00000000000000000000000000000000",
            "legacy",
            "legacy task",
            "pending",
            "[]",
            None,
            None,
            None,
            "2026-08-25T00:00:00+00:00",
            "2026-08-25T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    store = SQLiteTaskStore(database_path)
    try:
        legacy = store.get("task_00000000000000000000000000000000")
        created = store.create(
            "correlated",
            conversation_id="conversation-123",
            run_id="run-456",
        )
    finally:
        store.close()

    assert legacy.conversation_id is None
    assert legacy.run_id is None

    restarted = SQLiteTaskStore(database_path)
    try:
        restored = restarted.get(created.task_id)
    finally:
        restarted.close()
    assert restored.conversation_id == "conversation-123"
    assert restored.run_id == "run-456"
