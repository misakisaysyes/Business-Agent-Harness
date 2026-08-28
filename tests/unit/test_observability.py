"""请求级模型事件收集测试。"""

import asyncio
import json
import logging
from io import StringIO

import httpx
import structlog

from entrypoints.api import create_app
from harness.logging import AgentLog
from services.config import LogFormat, LoggingSettings, LogLevel
from services.model_gateway import ModelGatewayEvent, ModelGatewayEventType
from services.observability import (
    ConsoleLogFormatter,
    JsonLogFormatter,
    ModelGatewayEventCollector,
    configure_logging,
)


async def test_concurrent_run_events_are_isolated() -> None:
    """两个并发 Run 的模型网关事件不得交叉。"""

    collector = ModelGatewayEventCollector()
    ready = asyncio.Event()
    active = 0

    async def capture(label: str) -> list[ModelGatewayEvent]:
        nonlocal active
        with collector.capture() as events:
            active += 1
            if active == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            collector.emit(
                ModelGatewayEvent(
                    event_type=ModelGatewayEventType.RETRY,
                    model=label,
                    reason="TimeoutError",
                )
            )
            return events

    first, second = await asyncio.gather(capture("first"), capture("second"))

    assert [event.model for event in first] == ["first"]
    assert [event.model for event in second] == ["second"]


async def test_agent_server_assigns_a_trace_id_to_every_http_request() -> None:
    """HTTP 边界应创建 Trace ID 并返回给调用方用于日志关联。"""

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Trace-ID"].startswith("trace_")


class RecordingMonitorLog(AgentLog):
    """测试 record 已经接通未来线上监控出口。"""

    def __init__(self) -> None:
        super().__init__("tests.agent_log")
        self.events: list[tuple[str, dict[str, object]]] = []

    def monitor(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_agent_log_records_debug_context_and_calls_safe_monitor_placeholder() -> None:
    """统一 Log 应继承关联字段、过滤正文，并同时调用监控占位出口。"""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    stream = StringIO()
    observed = RecordingMonitorLog()
    try:
        configure_logging(
            LoggingSettings(level=LogLevel.DEBUG, format=LogFormat.JSON),
            stream=stream,
        )
        with observed.bind(
            trace_id="trace-1",
            conversation_id="conversation-1",
            thread_id="thread-1",
            run_id="run-1",
        ):
            observed.record(
                "rag.pipeline.finished",
                candidate_count=8,
                selected_count=3,
                query_text="must-not-leak",
            )
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)

    payload = json.loads(stream.getvalue())
    assert payload["event"] == "rag.pipeline.finished"
    assert payload["trace_id"] == "trace-1"
    assert payload["conversation_id"] == "conversation-1"
    assert payload["candidate_count"] == 8
    assert payload["selected_count"] == 3
    assert "must-not-leak" not in stream.getvalue()
    assert observed.events == [
        (
            "rag.pipeline.finished",
            {
                "trace_id": "trace-1",
                "conversation_id": "conversation-1",
                "thread_id": "thread-1",
                "run_id": "run-1",
                "candidate_count": 8,
                "selected_count": 3,
            },
        )
    ]


def test_agent_log_default_monitor_is_a_noop() -> None:
    """线上监控出口本阶段应可安全调用且不打印内容。"""

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("tests.monitor_noop")
    logger.addHandler(handler)
    try:
        AgentLog("tests.monitor_noop").monitor(
            "rag.pipeline.finished",
            selected_count=1,
        )
    finally:
        logger.removeHandler(handler)

    assert stream.getvalue() == ""


def log_record(**extra: object) -> logging.LogRecord:
    """创建包含测试字段的 LogRecord。"""

    record = logging.LogRecord(
        name="harness.hooks",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="tool call started",
        args=(),
        exc_info=None,
    )
    for name, value in extra.items():
        setattr(record, name, value)
    return record


def test_console_log_formatter_includes_correlation_without_sensitive_fields() -> None:
    """终端日志应包含关联字段且不输出未允许的敏感字段。"""

    output = ConsoleLogFormatter().format(
        log_record(
            thread_id="thread-1",
            run_id="run-1",
            tool_name="file_reader",
            tool_use_id="call-1",
            tool_input={"api_key": "secret-value"},
        )
    )

    assert "tool call started" in output
    assert "thread_id=thread-1" in output
    assert "run_id=run-1" in output
    assert "tool_name=file_reader" in output
    assert "tool_use_id=call-1" in output
    assert "secret-value" not in output


def test_json_log_formatter_emits_parseable_allowlisted_fields() -> None:
    """JSON 日志应可解析且只包含允许的结构化字段。"""

    output = JsonLogFormatter().format(
        log_record(
            thread_id="thread-2",
            tool_is_error=False,
            result_content="private-result",
        )
    )
    payload = json.loads(output)

    assert payload["event"] == "tool call started"
    assert payload["thread_id"] == "thread-2"
    assert payload["tool_is_error"] is False
    assert "result_content" not in payload
    assert "private-result" not in output


def test_configure_logging_bridges_structlog_to_server_stream() -> None:
    """统一初始化后 structlog 和标准 logging 应写入同一个 Server 输出流。"""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    stream = StringIO()
    try:
        configure_logging(
            LoggingSettings(level=LogLevel.INFO, format=LogFormat.CONSOLE),
            stream=stream,
        )
        logging.getLogger("harness.hooks").info(
            "tool call started",
            extra={"thread_id": "thread-3", "tool_name": "calculator"},
        )
        structlog.get_logger("services.mcp_tools").warning(
            "mcp_server_discovery_failed",
            server_name="demo",
            error_type="TimeoutError",
            api_key="must-not-leak",
        )
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)

    output = stream.getvalue()
    assert "tool_name=calculator" in output
    assert "server_name=demo" in output
    assert "error_type=TimeoutError" in output
    assert "must-not-leak" not in output


def test_configure_logging_suppresses_sensitive_network_sdk_debug_logs() -> None:
    """应用 DEBUG 不得打开可能包含 Prompt、请求体或供应商 Header 的 SDK 日志。"""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    sdk_loggers = [
        logging.getLogger(name)
        for name in ("openai._base_client", "httpcore2.http11", "httpx2")
    ]
    previous_sdk_levels = [logger.level for logger in sdk_loggers]
    stream = StringIO()
    try:
        configure_logging(
            LoggingSettings(level=LogLevel.DEBUG, format=LogFormat.CONSOLE),
            stream=stream,
        )
        sdk_loggers[0].debug("request body contains must-not-leak")
        sdk_loggers[1].debug("response headers contain must-not-leak")
        sdk_loggers[2].info("request URL contains must-not-leak")
        logging.getLogger("services.model_gateway").debug("agent.model.provider.started")
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)
        for logger, previous_level in zip(sdk_loggers, previous_sdk_levels, strict=True):
            logger.setLevel(previous_level)

    output = stream.getvalue()
    assert "agent.model.provider.started" in output
    assert "must-not-leak" not in output


def test_configure_logging_suppresses_raw_uvicorn_access_paths() -> None:
    """真实 URL 路径参数不得绕过 Middleware 的路由模板日志。"""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    access_logger = logging.getLogger("uvicorn.access")
    previous_access_level = access_logger.level
    stream = StringIO()
    try:
        configure_logging(
            LoggingSettings(level=LogLevel.DEBUG, format=LogFormat.CONSOLE),
            stream=stream,
        )
        access_logger.info(
            '127.0.0.1:50008 - "POST /users/alice/conversations/private-id/messages" 200'
        )
        logging.getLogger("entrypoints.api").debug(
            "agent.http.request.finished",
            extra={
                "http_route": "/users/{user_id}/conversations/{conversation_id}/messages",
                "http_status_code": 200,
                "trace_id": "trace-safe",
            },
        )
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)
        access_logger.setLevel(previous_access_level)

    output = stream.getvalue()
    assert "/users/{user_id}/conversations/{conversation_id}/messages" in output
    assert "trace-safe" in output
    assert "alice" not in output
    assert "private-id" not in output
