"""个人项目 ModelGateway 测试。

Tests for the personal-project model gateway.
"""

import asyncio
import threading
from collections.abc import Awaitable
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.messages import Message, MessageRole
from harness.model import ModelRequest
from harness.tool_use import ToolDefinition
from services.config import ModelGatewaySettings
from services.model_gateway import (
    ModelGateway,
    ModelGatewayEvent,
    ModelGatewayEventType,
    ModelGatewayRequestRejectedError,
    ModelGatewayUnavailableError,
    ModelRoute,
)


def request() -> ModelRequest:
    """创建网关测试使用的最小模型请求。

    Create the minimal model request used by gateway tests.
    """

    return ModelRequest(
        system_prompt="You are a test assistant.",
        messages=(Message(role=MessageRole.USER, content="test"),),
    )


class TimeoutThenSuccessModel:
    """先超时指定次数，随后成功的模型替身。

    Model double that times out a configured number of times before succeeding.
    """

    name = "timeout_model"

    def __init__(self, failures: int, content: str = "success") -> None:
        self.failures = failures
        self.response = Message(role=MessageRole.ASSISTANT, content=content)
        self.sync_calls = 0
        self.async_calls = 0

    def invoke(self, model_request: ModelRequest) -> Message:
        self.sync_calls += 1
        if self.sync_calls <= self.failures:
            raise TimeoutError("temporary timeout")
        return self.response

    async def ainvoke(self, model_request: ModelRequest) -> Message:
        self.async_calls += 1
        if self.async_calls <= self.failures:
            raise TimeoutError("temporary timeout")
        return self.response


class BlockingModel:
    """在测试释放信号前保持请求运行的模型替身。

    Model double keeping a request active until the test releases it.
    """

    name = "blocking"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.response = Message(role=MessageRole.ASSISTANT, content="primary")

    def invoke(self, model_request: ModelRequest) -> Message:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocking model")
        return self.response

    async def ainvoke(self, model_request: ModelRequest) -> Message:
        return await asyncio.to_thread(self.invoke, model_request)


class FixedModel:
    """返回固定响应并记录调用次数的模型替身。

    Model double returning a fixed response and recording call counts.
    """

    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.response = Message(role=MessageRole.ASSISTANT, content=content)
        self.sync_calls = 0
        self.async_calls = 0

    def invoke(self, model_request: ModelRequest) -> Message:
        self.sync_calls += 1
        return self.response

    async def ainvoke(self, model_request: ModelRequest) -> Message:
        self.async_calls += 1
        return self.response


class StatusError(RuntimeError):
    """携带 HTTP 状态码的模型错误。

    Model error carrying an HTTP status code.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RetryAfterStatusError(StatusError):
    """携带 Retry-After Header 的限流错误。"""

    def __init__(self, retry_after: str) -> None:
        super().__init__(429)
        self.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": retry_after},
        )


class ErrorModel:
    """每次调用都抛出固定错误的模型替身。

    Model double raising the same error for every invocation.
    """

    name = "error_model"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.sync_calls = 0
        self.async_calls = 0

    def invoke(self, model_request: ModelRequest) -> Message:
        self.sync_calls += 1
        raise self.error

    async def ainvoke(self, model_request: ModelRequest) -> Message:
        self.async_calls += 1
        raise self.error


def settings(lock_directory: Path, max_retries: int = 2) -> ModelGatewaySettings:
    """创建不实际等待的测试配置。

    Create gateway settings used by tests without real waiting.
    """

    return ModelGatewaySettings(
        max_retries=max_retries,
        retry_base_delay_seconds=1,
        retry_max_delay_seconds=16,
        retry_jitter_seconds=0.5,
        lock_directory=lock_directory,
    )


def immediate_async_sleep(delay: float) -> Awaitable[None]:
    """返回立即完成的异步等待。

    Return an asynchronous wait that completes immediately.
    """

    return asyncio.sleep(0)


def test_timeout_retries_two_times_with_exponential_backoff_and_jitter(
    tmp_path: Path,
) -> None:
    """超时应最多重试两次，并暴露指数退避和抖动信息。

    A timeout should retry twice and expose backoff and jitter information.
    """

    model = TimeoutThenSuccessModel(failures=2)
    events: list[ModelGatewayEvent] = []
    delays: list[float] = []
    gateway = ModelGateway(
        primary=ModelRoute(model, "primary", "retry-scope", 1),
        settings=settings(tmp_path),
        event_handler=events.append,
        sleeper=delays.append,
        jitter=lambda lower, upper: upper,
    )

    result = gateway.invoke(request())

    assert result.content == "success"
    assert model.sync_calls == 3
    assert delays == [1.5, 2.5]
    retry_events = [event for event in events if event.event_type is ModelGatewayEventType.RETRY]
    assert [event.retry_number for event in retry_events] == [1, 2]
    assert all(event.max_retries == 2 for event in retry_events)
    assert events[-1].event_type is ModelGatewayEventType.SELECTED


async def test_async_timeout_uses_the_same_retry_policy(tmp_path: Path) -> None:
    """异步模型调用应使用相同重试策略。

    Async model invocation should use the same retry policy.
    """

    model = TimeoutThenSuccessModel(failures=1, content="async success")
    events: list[ModelGatewayEvent] = []
    gateway = ModelGateway(
        primary=ModelRoute(model, "primary", "async-retry-scope", 1),
        settings=settings(tmp_path),
        event_handler=events.append,
        async_sleeper=immediate_async_sleep,
        jitter=lambda lower, upper: 0,
    )

    result = await gateway.ainvoke(request())

    assert result.content == "async success"
    assert model.async_calls == 2
    assert events[0].event_type is ModelGatewayEventType.RETRY
    assert events[0].retry_number == 1


def test_busy_primary_falls_back_to_available_model(tmp_path: Path) -> None:
    """主模型并发槽被占用时应立即选择可用备用模型。

    The gateway should select an available fallback when primary capacity is busy.
    """

    primary = BlockingModel()
    fallback = FixedModel("fallback", "fallback response")
    events: list[ModelGatewayEvent] = []
    gateway = ModelGateway(
        primary=ModelRoute(primary, "primary", "busy-primary-scope", 1),
        fallbacks=(ModelRoute(fallback, "fallback", "available-fallback-scope", 1),),
        settings=settings(tmp_path),
        event_handler=events.append,
        sleeper=lambda delay: None,
    )
    primary_results: list[Message] = []
    primary_thread = threading.Thread(
        target=lambda: primary_results.append(gateway.invoke(request())),
        daemon=True,
    )

    primary_thread.start()
    assert primary.started.wait(timeout=2)
    fallback_result = gateway.invoke(request())
    primary.release.set()
    primary_thread.join(timeout=2)

    assert fallback_result.content == "fallback response"
    assert fallback.sync_calls == 1
    assert primary_results[0].content == "primary"
    fallback_event = next(
        event for event in events if event.event_type is ModelGatewayEventType.FALLBACK
    )
    assert fallback_event.model == "blocking/primary"
    assert fallback_event.fallback_model == "fallback/fallback"
    assert "concurrency limit" in fallback_event.reason


def test_retry_exhaustion_falls_back_to_available_model(tmp_path: Path) -> None:
    """主模型耗尽临时错误重试后应选择备用模型。

    The gateway should use a fallback after primary transient retries are exhausted.
    """

    primary = TimeoutThenSuccessModel(failures=10)
    fallback = FixedModel("fallback", "degraded response")
    events: list[ModelGatewayEvent] = []
    gateway = ModelGateway(
        primary=ModelRoute(primary, "primary", "failed-primary-scope", 1),
        fallbacks=(ModelRoute(fallback, "fallback", "fallback-scope", 1),),
        settings=settings(tmp_path, max_retries=2),
        event_handler=events.append,
        sleeper=lambda delay: None,
        jitter=lambda lower, upper: 0,
    )

    result = gateway.invoke(request())

    assert result.content == "degraded response"
    assert primary.sync_calls == 3
    assert fallback.sync_calls == 1
    assert [event.event_type for event in events] == [
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.FALLBACK,
        ModelGatewayEventType.SELECTED,
    ]


def test_rate_limit_retries_once_before_falling_back(
    tmp_path: Path,
) -> None:
    """主模型返回 429 时应重试一次，仍失败再降级。

    A primary 429 should retry once before falling back to another model.
    """

    primary = ErrorModel(StatusError(429))
    fallback = FixedModel("deepseek", "fallback after 429")
    events: list[ModelGatewayEvent] = []
    delays: list[float] = []
    gateway = ModelGateway(
        primary=ModelRoute(primary, "kimi-k3", "moonshot-scope", 1),
        fallbacks=(ModelRoute(fallback, "deepseek-v4-flash", "deepseek-scope", 2),),
        settings=settings(tmp_path),
        event_handler=events.append,
        sleeper=delays.append,
        jitter=lambda lower, upper: upper,
    )

    result = gateway.invoke(request())

    assert result.content == "fallback after 429"
    assert primary.sync_calls == 2
    assert fallback.sync_calls == 1
    assert delays == [1.5]
    assert [event.event_type for event in events] == [
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.FALLBACK,
        ModelGatewayEventType.SELECTED,
    ]
    assert events[0].retry_number == 1
    assert events[0].max_retries == 1
    assert "rate-limit retries" in events[1].reason


def test_rate_limit_respects_retry_after_within_the_configured_cap(
    tmp_path: Path,
) -> None:
    """429 应优先遵守 Retry-After，同时受最大退避时间保护。"""

    model = ErrorModel(RetryAfterStatusError("7"))
    delays: list[float] = []
    gateway = ModelGateway(
        primary=ModelRoute(model, "primary", "retry-after-scope", 1),
        settings=settings(tmp_path),
        sleeper=delays.append,
        jitter=lambda lower, upper: 0,
    )

    with pytest.raises(ModelGatewayUnavailableError, match="rate-limit"):
        gateway.invoke(request())

    assert delays == [7.0]
    assert model.sync_calls == 2


def test_all_routes_report_failure_after_two_non_rate_limit_retries(
    tmp_path: Path,
) -> None:
    """主辅模型各重试两次仍失败时应返回明确的最终错误。"""

    primary = ErrorModel(TimeoutError("primary timeout"))
    fallback = ErrorModel(TimeoutError("fallback timeout"))
    events: list[ModelGatewayEvent] = []
    gateway = ModelGateway(
        primary=ModelRoute(primary, "primary", "failure-primary-scope", 1),
        fallbacks=(ModelRoute(fallback, "fallback", "failure-fallback-scope", 1),),
        settings=settings(tmp_path),
        event_handler=events.append,
        sleeper=lambda delay: None,
        jitter=lambda lower, upper: 0,
    )

    with pytest.raises(ModelGatewayUnavailableError, match="exhausted"):
        gateway.invoke(request())

    assert primary.sync_calls == 3
    assert fallback.sync_calls == 3
    assert [event.event_type for event in events] == [
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.FALLBACK,
        ModelGatewayEventType.RETRY,
        ModelGatewayEventType.RETRY,
    ]


def test_provider_bad_request_is_visible_and_not_retried(tmp_path: Path) -> None:
    """Provider 400 应转换为可见错误，且不得重试或降级。"""

    primary = ErrorModel(StatusError(400))
    fallback = FixedModel("fallback", "must not run")
    gateway = ModelGateway(
        primary=ModelRoute(primary, "primary", "bad-request-primary", 1),
        fallbacks=(ModelRoute(fallback, "fallback", "bad-request-fallback", 1),),
        settings=settings(tmp_path),
        sleeper=lambda delay: None,
    )

    with pytest.raises(ModelGatewayRequestRejectedError, match="HTTP 400"):
        gateway.invoke(request())

    assert primary.sync_calls == 1
    assert fallback.sync_calls == 0


def test_required_tool_skips_an_incompatible_fallback(tmp_path: Path) -> None:
    """强制工具调用不得降级到不支持 tool_choice 的 Thinking 模型。"""

    primary = ErrorModel(StatusError(429))
    incompatible_fallback = FixedModel("deepseek", "must not run")
    gateway = ModelGateway(
        primary=ModelRoute(primary, "kimi-k3", "required-primary", 1),
        fallbacks=(
            ModelRoute(
                incompatible_fallback,
                "deepseek-v4-flash",
                "required-fallback",
                1,
                supports_required_tool_choice=False,
            ),
        ),
        settings=settings(tmp_path),
        sleeper=lambda delay: None,
        jitter=lambda lower, upper: 0,
    )
    forced_request = ModelRequest(
        system_prompt="Use the required tool.",
        messages=(Message(role=MessageRole.USER, content="write"),),
        tools=(
            ToolDefinition(
                name="report_writer",
                description="Write a report.",
                parameters={"type": "object"},
            ),
        ),
        required_tool="report_writer",
    )

    with pytest.raises(ModelGatewayUnavailableError, match="does not support"):
        gateway.invoke(forced_request)

    assert primary.sync_calls == 2
    assert incompatible_fallback.sync_calls == 0
