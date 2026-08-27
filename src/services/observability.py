"""应用日志和请求级观测事件。

Application logging and request-scoped observability events.
"""

import json
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TextIO

import structlog

from services.config import LogFormat, LoggingSettings
from services.model_gateway import ModelGatewayEvent

SAFE_LOG_FIELDS = (
    "thread_id",
    "run_id",
    "tool_name",
    "tool_use_id",
    "tool_is_error",
    "tool_output_chars",
    "hook_name",
    "hook_event",
    "hook_error_type",
    "server_name",
    "error_type",
)


def _safe_log_fields(record: logging.LogRecord) -> dict[str, object]:
    """仅提取允许输出的结构化字段，避免意外记录参数和密钥。

    Extract allowlisted structured fields so arguments and secrets are not logged.
    """

    return {
        name: value
        for name in SAFE_LOG_FIELDS
        if (value := getattr(record, name, None)) is not None
    }


class ConsoleLogFormatter(logging.Formatter):
    """面向本地 Agent Server 终端的单行日志格式。"""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        )
        line = f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
        fields = _safe_log_fields(record)
        if fields:
            details = " ".join(f"{name}={value}" for name, value in fields.items())
            line = f"{line} {details}"
        if record.exc_info is not None and record.exc_info[0] is not None:
            line = f"{line} exception_type={record.exc_info[0].__name__}"
        return line


class JsonLogFormatter(logging.Formatter):
    """面向日志采集系统的单行 JSON 日志格式。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_safe_log_fields(record),
        }
        if record.exc_info is not None and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    settings: LoggingSettings,
    stream: TextIO | None = None,
) -> None:
    """为 Agent Server 初始化标准 logging 和 structlog。

    Initialize standard logging and structlog for the Agent Server process.
    """

    handler = logging.StreamHandler(stream or sys.stderr)
    formatter: logging.Formatter = (
        JsonLogFormatter() if settings.format is LogFormat.JSON else ConsoleLogFormatter()
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.level.value)

    # Uvicorn 默认会安装自己的 Handler。serve(log_config=None) 后让它统一向根日志传播。
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.render_to_log_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class ModelGatewayEventCollector:
    """按当前异步请求隔离模型重试和降级事件。

    Isolate model retry and fallback events by the current async request.
    """

    def __init__(self) -> None:
        self._events: ContextVar[list[ModelGatewayEvent] | None] = ContextVar(
            "model_gateway_events",
            default=None,
        )

    def emit(self, event: ModelGatewayEvent) -> None:
        """把事件写入当前请求的收集器；无活跃请求时忽略。

        Append an event to the current request collector, if one is active.
        """

        events = self._events.get()
        if events is not None:
            events.append(event)

    @contextmanager
    def capture(self) -> Generator[list[ModelGatewayEvent]]:
        """为当前上下文创建独立事件列表。

        Create an isolated event list for the current context.
        """

        events: list[ModelGatewayEvent] = []
        token = self._events.set(events)
        try:
            yield events
        finally:
            self._events.reset(token)


__all__ = [
    "ConsoleLogFormatter",
    "JsonLogFormatter",
    "ModelGatewayEventCollector",
    "configure_logging",
]
