"""Memory 和后续 Task Store 的本地实现。

Local implementations for memory and future task stores.
"""

import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import uuid4

import yaml

from harness.capabilities.memory import (
    DEFAULT_MAX_MEMORIES,
    MEMORY_INDEX_FILE,
    MemoryDraft,
    MemoryEntry,
    MemoryIndexEntry,
    MemoryType,
    memory_search_terms,
)
from harness.capabilities.task_system import (
    TaskDependencyError,
    TaskNotFoundError,
    TaskRecord,
    TaskStatus,
    TaskTransitionError,
)
from harness.conversation import ConversationRecord, ConversationStatus
from harness.permissions import PermissionRequest

MEMORY_HISTORY_DIRECTORY = ".history"
MAX_MEMORY_FILE_CHARACTERS = 50_000


class MemoryStoreError(RuntimeError):
    """本地 Memory Store 错误基类。"""


class MemoryCapacityError(MemoryStoreError):
    """Memory 数量达到配置上限。"""


class InvalidMemoryFileError(MemoryStoreError, ValueError):
    """磁盘上的 Memory 文件无效。"""


class FileMemoryStore:
    """按用户目录保存 Markdown Memory、索引和 JSONL 审计历史。"""

    def __init__(
        self,
        root: str | Path,
        max_memories: int = DEFAULT_MAX_MEMORIES,
    ) -> None:
        if max_memories < 1:
            raise ValueError("max_memories must be at least 1")
        self.root = Path(root).resolve()
        self.max_memories = max_memories
        self._lock = RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.history_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._lock:
            self._rebuild_index()

    @property
    def history_root(self) -> Path:
        return self.root / MEMORY_HISTORY_DIRECTORY

    @property
    def index_path(self) -> Path:
        return self.root / MEMORY_INDEX_FILE

    def upsert(self, draft: MemoryDraft, tool_use_id: str) -> MemoryEntry:
        """按 name 新建或更新，合并 tags 并把每个版本写入审计日志。"""

        with self._lock:
            existing = self.get(draft.name)
            if existing is None and len(self._memory_paths()) >= self.max_memories:
                raise MemoryCapacityError(
                    f"memory count exceeds configured maximum: {self.max_memories}"
                )

            now = datetime.now(UTC)
            tags = tuple(dict.fromkeys((*existing.tags, *draft.tags))) if existing else draft.tags
            entry = MemoryEntry(
                name=draft.name,
                memory_type=draft.memory_type,
                description=draft.description,
                content=draft.content,
                tags=tags,
                source=draft.source,
                created_at=existing.created_at if existing else now,
                updated_at=now,
                revision=existing.revision + 1 if existing else 1,
            )
            self._write_entry(entry)
            self._append_history(entry, tool_use_id, "updated" if existing else "created")
            self._rebuild_index()
            return entry

    def get(self, name: str) -> MemoryEntry | None:
        """读取一个当前版本；不存在时返回 None。"""

        path = self._memory_path(name)
        with self._lock:
            if not path.is_file():
                return None
            return self._read_entry(path)

    def list_index(self) -> tuple[MemoryIndexEntry, ...]:
        """按最近更新时间返回索引元数据。"""

        with self._lock:
            entries = [self._read_entry(path) for path in self._memory_paths()]
        entries.sort(key=lambda item: (-item.updated_at.timestamp(), item.name))
        return tuple(MemoryIndexEntry.from_entry(entry) for entry in entries)

    def search(
        self,
        query: str,
        memory_types: Sequence[MemoryType] = (),
        limit: int = 5,
    ) -> tuple[MemoryEntry, ...]:
        """使用确定性关键词评分选择相关 Memory，不额外调用模型。"""

        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_terms = memory_search_terms(query)
        if not query_terms:
            return ()
        allowed_types = frozenset(memory_types)

        with self._lock:
            entries = [self._read_entry(path) for path in self._memory_paths()]

        scored: list[tuple[int, float, MemoryEntry]] = []
        for entry in entries:
            if allowed_types and entry.memory_type not in allowed_types:
                continue
            identity_terms = memory_search_terms(f"{entry.name} {' '.join(entry.tags)}")
            description_terms = memory_search_terms(entry.description)
            content_terms = memory_search_terms(entry.content)
            score = (
                4 * len(query_terms & identity_terms)
                + 2 * len(query_terms & description_terms)
                + len(query_terms & content_terms)
            )
            if score > 0:
                scored.append((score, entry.updated_at.timestamp(), entry))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2].name))
        return tuple(item[2] for item in scored[:limit])

    def read_history(self, name: str) -> tuple[dict[str, Any], ...]:
        """读取测试和审计所需的完整版本事件。"""

        path = self.history_root / f"{name}.jsonl"
        with self._lock:
            if not path.is_file():
                return ()
            events: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    loaded = cast(object, json.loads(line))
                except json.JSONDecodeError as error:
                    raise InvalidMemoryFileError(f"invalid memory history: {path}") from error
                if not isinstance(loaded, dict):
                    raise InvalidMemoryFileError(f"memory history event must be an object: {path}")
                events.append(cast(dict[str, Any], loaded))
            return tuple(events)

    def _memory_path(self, name: str) -> Path:
        draft_name = MemoryDraft.model_validate(
            {
                "name": name,
                "memory_type": "reference",
                "description": "path validation",
                "content": "path validation",
                "source": "path validation",
            }
        ).name
        path = (self.root / f"{draft_name}.md").resolve()
        if not path.is_relative_to(self.root):
            raise InvalidMemoryFileError("memory path escapes configured root")
        return path

    def _memory_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                path
                for path in self.root.glob("*.md")
                if path.name != MEMORY_INDEX_FILE and path.resolve().is_relative_to(self.root)
            )
        )

    def _write_entry(self, entry: MemoryEntry) -> None:
        metadata = {
            "name": entry.name,
            "description": entry.description,
            "type": entry.memory_type.value,
            "tags": list(entry.tags),
            "source": entry.source,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "revision": entry.revision,
        }
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        self._atomic_write(
            self._memory_path(entry.name),
            f"---\n{frontmatter}\n---\n\n{entry.content.strip()}\n",
        )

    def _read_entry(self, path: Path) -> MemoryEntry:
        text = path.read_text(encoding="utf-8")
        if len(text) > MAX_MEMORY_FILE_CHARACTERS:
            raise InvalidMemoryFileError(f"memory file is too large: {path}")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise InvalidMemoryFileError(f"memory frontmatter is required: {path}")
        try:
            end = lines.index("---", 1)
        except ValueError as error:
            raise InvalidMemoryFileError(f"memory frontmatter is not closed: {path}") from error
        try:
            loaded = cast(object, yaml.safe_load("\n".join(lines[1:end])))
        except yaml.YAMLError as error:
            raise InvalidMemoryFileError(f"invalid memory frontmatter: {path}") from error
        if not isinstance(loaded, dict):
            raise InvalidMemoryFileError(f"memory frontmatter must be an object: {path}")
        metadata = cast(dict[str, Any], loaded)
        content = "\n".join(lines[end + 1 :]).strip()
        try:
            return MemoryEntry(
                name=metadata["name"],
                memory_type=metadata["type"],
                description=metadata["description"],
                content=content,
                tags=tuple(metadata.get("tags", ())),
                source=metadata["source"],
                created_at=metadata["created_at"],
                updated_at=metadata["updated_at"],
                revision=metadata["revision"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidMemoryFileError(f"invalid memory metadata: {path}: {error}") from error

    def _append_history(
        self,
        entry: MemoryEntry,
        tool_use_id: str,
        operation: str,
    ) -> None:
        event = {
            "operation": operation,
            "tool_use_id": tool_use_id,
            "entry": entry.model_dump(mode="json"),
        }
        path = self.history_root / f"{entry.name}.jsonl"
        with path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        path.chmod(0o600)

    def _rebuild_index(self) -> None:
        entries = [self._read_entry(path) for path in self._memory_paths()]
        entries.sort(key=lambda item: item.name)
        lines = ["# Memory Index", ""]
        lines.extend(
            f"- [{entry.name}]({entry.name}.md) — {entry.description} "
            f"[{entry.memory_type.value}; revision {entry.revision}]"
            for entry in entries
        )
        self._atomic_write(self.index_path, "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


class SQLiteTaskStore:
    """使用事务保证依赖检查和单执行者认领的用户级 Task Store。

    User-scoped task store with transactional dependency checks and single-owner claims.
    """

    def __init__(self, database_path: str | Path, busy_timeout_seconds: float = 5.0) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1_000)}"
        )
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                dependencies TEXT NOT NULL,
                owner TEXT,
                result_reference TEXT,
                failure_reason TEXT,
                conversation_id TEXT,
                run_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_task_correlation_columns()
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS tasks_status ON tasks (status, created_at)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS tasks_conversation "
            "ON tasks (conversation_id, created_at)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS tasks_run ON tasks (run_id, created_at)"
        )
        self.database_path.chmod(0o600)

    def create(
        self,
        title: str,
        description: str = "",
        dependencies: Sequence[str] = (),
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> TaskRecord:
        """创建 Pending Task，并要求所有依赖已经存在。"""

        normalized_dependencies = tuple(dependencies)
        if len(normalized_dependencies) != len(set(normalized_dependencies)):
            raise TaskDependencyError("task dependencies must be unique")

        with self._transaction():
            missing = [
                dependency
                for dependency in normalized_dependencies
                if self._select_task(dependency) is None
            ]
            if missing:
                raise TaskDependencyError(
                    f"unknown task dependencies: {', '.join(missing)}"
                )
            now = datetime.now(UTC)
            task = TaskRecord(
                task_id=f"task_{uuid4().hex}",
                title=title,
                description=description,
                dependencies=normalized_dependencies,
                conversation_id=conversation_id,
                run_id=run_id,
                created_at=now,
                updated_at=now,
            )
            self._connection.execute(
                """
                INSERT INTO tasks (
                    task_id, title, description, status, dependencies, owner,
                    result_reference, failure_reason, conversation_id, run_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._task_values(task),
            )
            return task

    def get(self, task_id: str) -> TaskRecord:
        """读取一个 Task，不存在时返回明确领域错误。"""

        with self._lock:
            task = self._select_task(task_id)
        if task is None:
            raise TaskNotFoundError(f"unknown task: {task_id}")
        return task

    def list(self, status: TaskStatus | None = None) -> tuple[TaskRecord, ...]:
        """按创建顺序列出全部 Task 或指定状态的 Task。"""

        with self._lock:
            if status is None:
                rows = self._connection.execute(
                    "SELECT * FROM tasks ORDER BY created_at, task_id"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY created_at, task_id",
                    (status.value,),
                ).fetchall()
        return tuple(self._row_to_task(row) for row in rows)

    def claim(self, task_id: str, owner: str) -> TaskRecord:
        """原子认领依赖已完成的 Pending Task。"""

        with self._transaction():
            task = self._require_task(task_id)
            if task.status is not TaskStatus.PENDING:
                raise TaskTransitionError(f"task is not pending: {task_id}")
            blocked = [
                dependency
                for dependency in task.dependencies
                if self._require_task(dependency).status is not TaskStatus.COMPLETED
            ]
            if blocked:
                raise TaskDependencyError(
                    f"task dependencies are not completed: {', '.join(blocked)}"
                )
            updated_at = datetime.now(UTC)
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET status = ?, owner = ?, updated_at = ?
                WHERE task_id = ? AND status = ?
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    owner,
                    updated_at.isoformat(),
                    task_id,
                    TaskStatus.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise TaskTransitionError(f"task was already claimed: {task_id}")
            return self._require_task(task_id)

    def complete(self, task_id: str, owner: str, result_reference: str) -> TaskRecord:
        """把 Owner 持有的 In-Progress Task 完成为终态。"""

        return self._finish(
            task_id,
            owner,
            TaskStatus.COMPLETED,
            result_reference=result_reference,
        )

    def fail(self, task_id: str, owner: str, failure_reason: str) -> TaskRecord:
        """把 Owner 持有的 In-Progress Task 标记为失败终态。"""

        return self._finish(
            task_id,
            owner,
            TaskStatus.FAILED,
            failure_reason=failure_reason,
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _finish(
        self,
        task_id: str,
        owner: str,
        status: TaskStatus,
        result_reference: str | None = None,
        failure_reason: str | None = None,
    ) -> TaskRecord:
        with self._transaction():
            task = self._require_task(task_id)
            if task.status is not TaskStatus.IN_PROGRESS:
                raise TaskTransitionError(f"task is not in_progress: {task_id}")
            if task.owner != owner:
                raise TaskTransitionError(
                    f"task owner mismatch: expected {task.owner}, got {owner}"
                )
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET status = ?, result_reference = ?, failure_reason = ?, updated_at = ?
                WHERE task_id = ? AND status = ? AND owner = ?
                """,
                (
                    status.value,
                    result_reference,
                    failure_reason,
                    datetime.now(UTC).isoformat(),
                    task_id,
                    TaskStatus.IN_PROGRESS.value,
                    owner,
                ),
            )
            if cursor.rowcount != 1:
                raise TaskTransitionError(f"task changed while finishing: {task_id}")
            return self._require_task(task_id)

    def _require_task(self, task_id: str) -> TaskRecord:
        task = self._select_task(task_id)
        if task is None:
            raise TaskNotFoundError(f"unknown task: {task_id}")
        return task

    def _select_task(self, task_id: str) -> TaskRecord | None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return self._row_to_task(row) if row is not None else None

    def _ensure_task_correlation_columns(self) -> None:
        """为旧版 Task 数据库增加可空关联列。

        Add nullable correlation columns to legacy task databases.
        """

        columns = {
            cast(str, row["name"])
            for row in self._connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        for column in ("conversation_id", "run_id"):
            if column in columns:
                continue
            try:
                self._connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError as error:
                if "duplicate column name" not in str(error).lower():
                    raise

    @staticmethod
    def _task_values(task: TaskRecord) -> tuple[object, ...]:
        return (
            task.task_id,
            task.title,
            task.description,
            task.status.value,
            json.dumps(task.dependencies),
            task.owner,
            task.result_reference,
            task.failure_reason,
            task.conversation_id,
            task.run_id,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TaskRecord:
        dependencies = cast(list[str], json.loads(cast(str, row["dependencies"])))
        return TaskRecord.model_validate(
            {
                "task_id": row["task_id"],
                "title": row["title"],
                "description": row["description"],
                "status": row["status"],
                "dependencies": tuple(dependencies),
                "owner": row["owner"],
                "result_reference": row["result_reference"],
                "failure_reason": row["failure_reason"],
                "conversation_id": row["conversation_id"],
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    class _Transaction:
        def __init__(self, store: "SQLiteTaskStore") -> None:
            self.store = store

        def __enter__(self) -> None:
            self.store._lock.acquire()
            try:
                self.store._connection.execute("BEGIN IMMEDIATE")
            except BaseException:
                self.store._lock.release()
                raise

        def __exit__(self, error_type: object, error: object, traceback: object) -> None:
            try:
                self.store._connection.execute("ROLLBACK" if error_type else "COMMIT")
            finally:
                self.store._lock.release()

    def _transaction(self) -> _Transaction:
        return self._Transaction(self)


class SQLiteConversationStore:
    """把 Conversation 所有权和恢复信息保存到本地 SQLite。

    Persist conversation ownership and recovery metadata in local SQLite.
    """

    def __init__(self, database_path: str | Path, busy_timeout_seconds: float = 5.0) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1_000)}"
        )
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_run_id TEXT,
                    permission_request TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_user_id
                ON conversations (user_id, conversation_id)
                """
            )
        self.database_path.chmod(0o600)

    def create(self, record: ConversationRecord) -> None:
        """新增一条 Conversation 记录。"""

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, user_id, status, active_run_id, permission_request
                ) VALUES (?, ?, ?, ?, ?)
                """,
                self._record_values(record),
            )

    def get(self, conversation_id: str) -> ConversationRecord | None:
        """按稳定 Conversation ID 查询记录。"""

        with self._lock:
            row = self._connection.execute(
                """
                SELECT conversation_id, user_id, status, active_run_id, permission_request
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list(self, user_id: str) -> tuple[ConversationRecord, ...]:
        """列出一个用户拥有的全部 Conversation。"""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT conversation_id, user_id, status, active_run_id, permission_request
                FROM conversations
                WHERE user_id = ?
                ORDER BY rowid
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._row_to_record(row) for row in rows)

    def save(self, record: ConversationRecord) -> None:
        """保存 Run 状态和待审批信息。"""

        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE conversations
                SET user_id = ?, status = ?, active_run_id = ?, permission_request = ?
                WHERE conversation_id = ?
                """,
                (
                    record.user_id,
                    record.status.value,
                    record.active_run_id,
                    self._serialize_permission(record.permission_request),
                    record.conversation_id,
                ),
            )

    def delete(self, conversation_id: str) -> None:
        """删除 Conversation 索引；Checkpoint 由 AgentLoop 单独清理。"""

        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

    def recover_running(self) -> None:
        """释放进程异常退出时遗留的普通 RUNNING 标记。

        Release normal RUNNING markers left behind by an abnormal process exit.
        """

        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE conversations
                SET status = ?, active_run_id = NULL, permission_request = NULL
                WHERE status = ?
                """,
                (ConversationStatus.IDLE.value, ConversationStatus.RUNNING.value),
            )

    def close(self) -> None:
        """关闭 Store 拥有的数据库连接。"""

        with self._lock:
            self._connection.close()

    @staticmethod
    def _serialize_permission(request: PermissionRequest | None) -> str | None:
        return request.model_dump_json() if request is not None else None

    @classmethod
    def _record_values(
        cls,
        record: ConversationRecord,
    ) -> tuple[str, str, str, str | None, str | None]:
        return (
            record.conversation_id,
            record.user_id,
            record.status.value,
            record.active_run_id,
            cls._serialize_permission(record.permission_request),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ConversationRecord:
        raw_request = cast(str | None, row["permission_request"])
        request = PermissionRequest.model_validate_json(raw_request) if raw_request else None
        return ConversationRecord(
            conversation_id=cast(str, row["conversation_id"]),
            user_id=cast(str, row["user_id"]),
            status=ConversationStatus(cast(str, row["status"])),
            active_run_id=cast(str | None, row["active_run_id"]),
            permission_request=request,
        )


__all__ = [
    "FileMemoryStore",
    "InvalidMemoryFileError",
    "MAX_MEMORY_FILE_CHARACTERS",
    "MEMORY_HISTORY_DIRECTORY",
    "MemoryCapacityError",
    "MemoryStoreError",
    "SQLiteConversationStore",
    "SQLiteTaskStore",
]
