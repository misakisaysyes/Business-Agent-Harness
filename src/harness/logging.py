"""Agent 与 RAG 共用的安全调试日志和监控门面。

Shared safe debug logging and monitoring facade for Agent and RAG flows.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

type LogValue = str | int | float | bool | None
type LogFields = dict[str, LogValue]

# 只允许输出运行标识、计数、耗时和技术配置。用户输入、Prompt、Tool 参数、
# 文档正文、向量和凭据即使被误传给 AgentLog 也不会进入日志或未来监控实现。
SAFE_LOG_FIELDS = frozenset(
    {
        "agent_name",
        "budget_dropped_count",
        "cancel_requested",
        "candidate_count",
        "chunk_count",
        "collection_name",
        "component",
        "context_characters",
        "conversation_id",
        "delay_seconds",
        "deleted_chunks",
        "document_count",
        "document_id",
        "duplicate_count",
        "duration_ms",
        "embedding_dimension",
        "embedding_model",
        "error_type",
        "failed_count",
        "fallback_model",
        "filter_count",
        "has_user_scope",
        "hook_error_type",
        "hook_event",
        "hook_name",
        "http_method",
        "http_route",
        "http_status_code",
        "include_public",
        "indexed_count",
        "iteration_count",
        "max_context_characters",
        "max_iterations",
        "max_output_tokens",
        "max_retries",
        "message_count",
        "model",
        "model_name",
        "operation",
        "permission_allow_count",
        "permission_ask_count",
        "permission_deny_count",
        "public_candidate_count",
        "query_characters",
        "rebuild",
        "remaining_after_top_k_count",
        "required_tool",
        "response_has_content",
        "response_tool_use_count",
        "retrieved_count",
        "role",
        "retry_number",
        "run_id",
        "scope",
        "score_threshold",
        "search_mode",
        "selected_count",
        "server_name",
        "skipped_count",
        "status",
        "stop_reason",
        "team_run_id",
        "task_id",
        "parent_task_id",
        "attempt",
        "attempts",
        "last_error",
        "allowed_tool_names",
        "threshold_dropped_count",
        "thread_id",
        "tool_count",
        "tool_definition_count",
        "tool_is_error",
        "tool_name",
        "tool_output_chars",
        "tool_use_id",
        "top_k",
        "trace_id",
        "user_candidate_count",
        "vector_count",
    }
)

_LOG_CONTEXT: ContextVar[LogFields | None] = ContextVar(
    "agent_log_context",
    default=None,
)


def _current_context() -> LogFields:
    return _LOG_CONTEXT.get() or {}


def new_trace_id() -> str:
    """生成不含用户信息的请求级 Trace ID。"""

    return f"trace_{uuid4().hex}"


def _safe_fields(fields: dict[str, object]) -> LogFields:
    """过滤未知字段和值类型，避免正文或复杂对象意外进入日志。"""

    safe: LogFields = {}
    for name, value in fields.items():
        if name not in SAFE_LOG_FIELDS:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[name] = value
    return safe


class AgentLog:
    """统一发出本地调试日志，并预留线上监控出口。

    ``record`` 是业务调用入口。它会先打印 DEBUG 日志，再调用目前为空实现的
    ``monitor``。未来接入 OpenTelemetry、LangSmith 或指标系统时，只需实现
    ``monitor``，现有 Agent/RAG 调用点无需再次修改。
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @staticmethod
    def context_fields() -> LogFields:
        """返回当前异步请求绑定的安全关联字段副本。"""

        return dict(_current_context())

    @contextmanager
    def bind(self, **fields: object) -> Generator[None]:
        """在当前同步/异步上下文中绑定 Trace、Conversation 和 Run 等字段。"""

        merged = {**_current_context(), **_safe_fields(fields)}
        token = _LOG_CONTEXT.set(merged)
        try:
            yield
        finally:
            _LOG_CONTEXT.reset(token)

    def debug(self, event: str, **fields: object) -> None:
        """向 Agent Server 的现有日志流打印一条安全 DEBUG 事件。"""

        self._logger.debug(event, extra=self._event_fields(fields))

    def monitor(self, event: str, **fields: object) -> None:
        """预留线上监控出口；本阶段不发送任何数据。"""

        del event, fields

    def record(self, event: str, **fields: object) -> None:
        """记录调试事件并调用线上监控占位接口。"""

        safe = self._event_fields(fields)
        self._logger.debug(event, extra=safe)
        self.monitor(event, **safe)

    def warning(self, event: str, **fields: object) -> None:
        """记录不含正文的安全告警，并预留同名监控事件。"""

        safe = self._event_fields(fields)
        self._logger.warning(event, extra=safe)
        self.monitor(event, **safe)

    def error(self, event: str, **fields: object) -> None:
        """记录不含异常正文的安全错误，并预留同名监控事件。"""

        safe = self._event_fields(fields)
        self._logger.error(event, extra=safe)
        self.monitor(event, **safe)

    @contextmanager
    def operation(self, event: str, **fields: object) -> Generator[LogFields]:
        """记录一个操作的 started/finished/failed 事件和耗时。

        调用方可以向返回的字典补充结果计数；这些字段仍会经过白名单过滤。
        """

        started = perf_counter()
        self.record(f"{event}.started", **fields)
        result_fields: LogFields = {}
        try:
            yield result_fields
        except BaseException as error:
            failed_fields = {
                **fields,
                **result_fields,
                "status": "error",
                "error_type": type(error).__name__,
                "duration_ms": round((perf_counter() - started) * 1_000, 3),
            }
            self.record(
                f"{event}.failed",
                **failed_fields,
            )
            raise
        else:
            completed_fields = {
                **fields,
                **result_fields,
                "status": result_fields.get("status", "ok"),
                "duration_ms": round((perf_counter() - started) * 1_000, 3),
            }
            self.record(
                f"{event}.finished",
                **completed_fields,
            )

    @staticmethod
    def _event_fields(fields: dict[str, object]) -> LogFields:
        return {**_current_context(), **_safe_fields(fields)}


__all__ = ["AgentLog", "LogFields", "LogValue", "SAFE_LOG_FIELDS", "new_trace_id"]
