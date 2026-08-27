"""LangGraph Checkpoint 的本地持久化适配器。

Local persistence adapters for LangGraph checkpoints.
"""

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from harness.messages import Message, MessageRole, ToolResult, ToolUse
from harness.state import AgentStopReason


def _create_serializer() -> JsonPlusSerializer:
    """创建只允许项目状态类型的序列化器。

    Create a serializer allowlisting only project state types.
    """

    return JsonPlusSerializer(
        allowed_msgpack_modules=(
            Message,
            MessageRole,
            ToolResult,
            ToolUse,
            AgentStopReason,
        )
    )


class LocalSqliteSaver(SqliteSaver):
    """同时支持同步和异步 AgentLoop 的本地 SQLite Saver。

    Local SQLite saver supporting both synchronous and asynchronous agent loops.

    官方 ``SqliteSaver`` 只提供同步数据库方法。本项目的 CLI 测试会使用
    同步 Loop，而 HTTP Server 使用异步 Loop，因此异步方法通过工作线程调用
    同一个带锁的 Saver，避免维护两套 Checkpoint 数据库。

    The official ``SqliteSaver`` exposes synchronous database methods only. The
    project uses a synchronous loop in local tests and an asynchronous loop in the
    HTTP server, so async methods delegate to the same locked saver in a worker
    thread instead of maintaining two checkpoint databases.
    """

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        checkpoints = await asyncio.to_thread(
            lambda: tuple(
                self.list(
                    config,
                    filter=filter,
                    before=before,
                    limit=limit,
                )
            )
        )
        for checkpoint in checkpoints:
            yield checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    def close(self) -> None:
        """关闭此 Saver 拥有的 SQLite 连接。"""

        with self.lock:
            self.conn.close()


def create_sqlite_checkpointer(
    database_path: str | Path,
    busy_timeout_seconds: float = 5.0,
) -> LocalSqliteSaver:
    """创建适合单机 CLI/Server 的持久化 SQLite Checkpointer。

    Create a persistent SQLite checkpointer for a local CLI/server process.
    """

    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1_000)}")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")

    saver = LocalSqliteSaver(connection, serde=_create_serializer())
    saver.setup()
    path.chmod(0o600)
    return saver


def create_in_memory_checkpointer() -> BaseCheckpointSaver[str]:
    """创建仅用于单元测试的进程内 Checkpointer。

    Create an in-memory checkpointer intended only for unit tests.
    """

    return InMemorySaver(serde=_create_serializer())


__all__ = [
    "LocalSqliteSaver",
    "create_in_memory_checkpointer",
    "create_sqlite_checkpointer",
]
