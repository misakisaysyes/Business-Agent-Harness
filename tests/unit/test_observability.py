"""请求级模型事件收集测试。"""

import asyncio
import json
import logging
from io import StringIO

import structlog

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
