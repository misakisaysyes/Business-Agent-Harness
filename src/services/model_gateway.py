"""个人项目使用的模型并发、降级和重试入口。

Model concurrency, fallback, and retry entry point for a personal project.
"""

import asyncio
import fcntl
import hashlib
import logging
import os
import random
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from harness.error_recovery import (
    ErrorKind,
    classify_error,
    has_status_code,
    is_retryable_model_error,
    retry_after_seconds,
    safe_error_reason,
    status_code,
)
from harness.messages import Message
from harness.model import ModelProvider, ModelRequest
from services.config import ModelGatewaySettings

logger = logging.getLogger(__name__)


class ModelGatewayEventType(StrEnum):
    """用户可见的模型网关事件类型。

    User-visible model-gateway event types.
    """

    RETRY = "retry"
    FALLBACK = "fallback"
    SELECTED = "selected"


@dataclass(frozen=True, slots=True)
class ModelGatewayEvent:
    """一次重试或模型降级事件。

    One retry or model-fallback event.
    """

    event_type: ModelGatewayEventType
    model: str
    reason: str
    retry_number: int | None = None
    max_retries: int | None = None
    delay_seconds: float | None = None
    fallback_model: str | None = None


ModelGatewayEventHandler = Callable[[ModelGatewayEvent], None]


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """一个可被 ModelGateway 选择的模型路由。

    One model route selectable by the model gateway.
    """

    provider: ModelProvider
    model_id: str
    quota_scope: str
    max_concurrency: int = 1
    supports_required_tool_choice: bool = True

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model route requires a model ID")
        if not self.quota_scope:
            raise ValueError("model route requires a quota scope")
        if self.max_concurrency < 1:
            raise ValueError("model route max concurrency must be at least 1")

    @property
    def label(self) -> str:
        """返回供日志和 CLI 展示的模型名称。

        Return the model name shown in logs and the CLI.
        """

        return f"{self.provider.name}/{self.model_id}"


class ModelGatewayUnavailableError(RuntimeError):
    """所有模型路由都因满载或临时故障而不可用。

    Raised when all model routes are unavailable due to capacity or transient failures.
    """


class ModelGatewayRequestRejectedError(RuntimeError):
    """模型服务拒绝了不可重试的请求参数。

    Raised when a model provider rejects non-retryable request parameters.
    """


@dataclass(slots=True)
class _SlotLease:
    """一个由当前进程持有的跨进程并发槽。

    One cross-process concurrency slot held by the current process.
    """

    pool: "_ConcurrencyPool"
    slot_index: int
    file_descriptor: int
    released: bool = False

    def release(self) -> None:
        """释放文件锁和本进程内的槽位标记。

        Release the file lock and the in-process slot marker.
        """

        if self.released:
            return
        self.released = True
        self.pool.release(self)


class _ConcurrencyPool:
    """使用多个文件锁实现指定额度范围的跨进程并发槽。

    Cross-process concurrency slots backed by file locks for one quota scope.
    """

    def __init__(self, scope: str, max_concurrency: int, lock_directory: Path) -> None:
        self.scope = scope
        self.max_concurrency = max_concurrency
        self.lock_directory = lock_directory.expanduser().resolve()
        self.lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._active_slots: set[int] = set()
        self._mutex = threading.Lock()
        scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
        self._slot_paths = tuple(
            self.lock_directory / f"model-{scope_hash}-{index}.lock"
            for index in range(max_concurrency)
        )

    def try_acquire(self) -> _SlotLease | None:
        """非阻塞获取一个并发槽；全部占用时返回空值。

        Acquire one slot without blocking, or return no value when all are busy.
        """

        with self._mutex:
            for slot_index, path in enumerate(self._slot_paths):
                if slot_index in self._active_slots:
                    continue

                file_descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(file_descriptor)
                    continue

                self._active_slots.add(slot_index)
                return _SlotLease(self, slot_index, file_descriptor)

        return None

    def release(self, lease: _SlotLease) -> None:
        """释放指定槽位。

        Release the specified slot.
        """

        try:
            fcntl.flock(lease.file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lease.file_descriptor)
            with self._mutex:
                self._active_slots.discard(lease.slot_index)


class _RouteAtCapacityError(RuntimeError):
    """一个模型路由当前没有可用并发槽。"""


class _RouteRetriesExhaustedError(RuntimeError):
    """一个模型路由已经耗尽临时错误重试次数。"""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class ModelGateway:
    """为 ModelProvider 增加跨进程并发、重试和降级能力。

    Add cross-process concurrency, retry, and fallback behavior to model providers.
    """

    name = "model_gateway"

    def __init__(
        self,
        primary: ModelRoute,
        fallbacks: Sequence[ModelRoute] = (),
        settings: ModelGatewaySettings | None = None,
        event_handler: ModelGatewayEventHandler | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        async_sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.routes = (primary, *fallbacks)
        self.settings = settings or ModelGatewaySettings()
        self.event_handler = event_handler
        self._sleep = sleeper
        self._async_sleep = async_sleeper
        self._jitter = jitter
        self._pools: dict[str, _ConcurrencyPool] = {}
        pool_limits: dict[str, int] = {}

        for route in self.routes:
            existing_limit = pool_limits.get(route.quota_scope)
            if existing_limit is not None and existing_limit != route.max_concurrency:
                raise ValueError("routes sharing a quota scope must use the same max concurrency")
            pool_limits[route.quota_scope] = route.max_concurrency
            self._pools.setdefault(
                route.quota_scope,
                _ConcurrencyPool(
                    route.quota_scope,
                    route.max_concurrency,
                    self.settings.lock_directory,
                ),
            )

    def invoke(self, request: ModelRequest) -> Message:
        """同步选择路由，并执行带重试的模型调用。

        Select a route synchronously and invoke the model with retries.
        """

        fallback_reason: str | None = None
        last_error: Exception | None = None

        for route_index, route in enumerate(self.routes):
            if request.required_tool is not None and not route.supports_required_tool_choice:
                fallback_reason = (
                    f"{route.label} does not support required tool choice"
                )
                continue
            try:
                return self._invoke_route(
                    route,
                    request,
                    fallback_from=self.routes[route_index - 1] if route_index else None,
                    fallback_reason=fallback_reason,
                )
            except _RouteAtCapacityError:
                fallback_reason = f"{route.label} reached its concurrency limit"
            except _RouteRetriesExhaustedError as error:
                last_error = error.error
                fallback_reason = self._retry_exhaustion_reason(route, error.error)

        raise self._unavailable_error(fallback_reason, last_error)

    async def ainvoke(self, request: ModelRequest) -> Message:
        """异步选择路由，并执行带重试的模型调用。

        Select a route asynchronously and invoke the model with retries.
        """

        fallback_reason: str | None = None
        last_error: Exception | None = None

        for route_index, route in enumerate(self.routes):
            if request.required_tool is not None and not route.supports_required_tool_choice:
                fallback_reason = (
                    f"{route.label} does not support required tool choice"
                )
                continue
            try:
                return await self._ainvoke_route(
                    route,
                    request,
                    fallback_from=self.routes[route_index - 1] if route_index else None,
                    fallback_reason=fallback_reason,
                )
            except _RouteAtCapacityError:
                fallback_reason = f"{route.label} reached its concurrency limit"
            except _RouteRetriesExhaustedError as error:
                last_error = error.error
                fallback_reason = self._retry_exhaustion_reason(route, error.error)

        raise self._unavailable_error(fallback_reason, last_error)

    def _invoke_route(
        self,
        route: ModelRoute,
        request: ModelRequest,
        fallback_from: ModelRoute | None,
        fallback_reason: str | None,
    ) -> Message:
        fallback_announced = False

        retry_number = 0
        while True:
            lease = self._pools[route.quota_scope].try_acquire()
            if lease is None:
                raise _RouteAtCapacityError(route.label)

            if fallback_from is not None and not fallback_announced:
                self._emit_fallback(fallback_from, route, fallback_reason)
                fallback_announced = True

            try:
                response = route.provider.invoke(request)
                self._emit_selected(route)
                return response
            except Exception as error:
                if not is_retryable_model_error(error):
                    self._raise_if_provider_rejected(error)
                    raise
                is_rate_limit = has_status_code(error, 429)
                retry_limit = (
                    self.settings.rate_limit_max_retries
                    if is_rate_limit
                    else self.settings.max_retries
                )
                if retry_number >= retry_limit:
                    raise _RouteRetriesExhaustedError(error) from error
                retry_number += 1
                delay = self._retry_delay(retry_number, error)
                self._emit_retry(route, error, retry_number, retry_limit, delay)
            finally:
                lease.release()

            self._sleep(delay)

    async def _ainvoke_route(
        self,
        route: ModelRoute,
        request: ModelRequest,
        fallback_from: ModelRoute | None,
        fallback_reason: str | None,
    ) -> Message:
        fallback_announced = False

        retry_number = 0
        while True:
            lease = self._pools[route.quota_scope].try_acquire()
            if lease is None:
                raise _RouteAtCapacityError(route.label)

            if fallback_from is not None and not fallback_announced:
                self._emit_fallback(fallback_from, route, fallback_reason)
                fallback_announced = True

            try:
                response = await route.provider.ainvoke(request)
                self._emit_selected(route)
                return response
            except Exception as error:
                if not is_retryable_model_error(error):
                    self._raise_if_provider_rejected(error)
                    raise
                is_rate_limit = has_status_code(error, 429)
                retry_limit = (
                    self.settings.rate_limit_max_retries
                    if is_rate_limit
                    else self.settings.max_retries
                )
                if retry_number >= retry_limit:
                    raise _RouteRetriesExhaustedError(error) from error
                retry_number += 1
                delay = self._retry_delay(retry_number, error)
                self._emit_retry(route, error, retry_number, retry_limit, delay)
            finally:
                lease.release()

            await self._async_sleep(delay)

    def _retry_delay(self, retry_number: int, error: Exception) -> float:
        exponential = self.settings.retry_base_delay_seconds * (2 ** (retry_number - 1))
        jitter = self._jitter(0.0, self.settings.retry_jitter_seconds)
        provider_delay = retry_after_seconds(error) or 0.0
        return min(
            self.settings.retry_max_delay_seconds,
            max(exponential + jitter, provider_delay),
        )

    def _emit_retry(
        self,
        route: ModelRoute,
        error: Exception,
        retry_number: int,
        retry_limit: int,
        delay: float,
    ) -> None:
        self._emit(
            ModelGatewayEvent(
                event_type=ModelGatewayEventType.RETRY,
                model=route.label,
                reason=safe_error_reason(error),
                retry_number=retry_number,
                max_retries=retry_limit,
                delay_seconds=delay,
            )
        )

    def _emit_fallback(
        self,
        previous: ModelRoute,
        fallback: ModelRoute,
        reason: str | None,
    ) -> None:
        self._emit(
            ModelGatewayEvent(
                event_type=ModelGatewayEventType.FALLBACK,
                model=previous.label,
                fallback_model=fallback.label,
                reason=reason or "primary model is unavailable",
            )
        )

    def _emit_selected(self, route: ModelRoute) -> None:
        """记录本次模型调用实际使用的成功路由。

        Record the successful route actually used for this model call.
        """

        self._emit(
            ModelGatewayEvent(
                event_type=ModelGatewayEventType.SELECTED,
                model=route.label,
                reason="model request succeeded",
            )
        )

    def _emit(self, event: ModelGatewayEvent) -> None:
        if self.event_handler is None:
            return
        try:
            self.event_handler(event)
        except Exception:
            logger.exception("model gateway event handler failed")

    @staticmethod
    def _raise_if_provider_rejected(error: Exception) -> None:
        response_status = status_code(error)
        if response_status is None or not 400 <= response_status <= 499:
            return
        raise ModelGatewayRequestRejectedError(
            f"model provider rejected request: {safe_error_reason(error)}"
        ) from error

    @staticmethod
    def _retry_exhaustion_reason(route: ModelRoute, error: Exception) -> str:
        kind = classify_error(error)
        if kind is ErrorKind.RATE_LIMIT:
            return f"{route.label} exhausted rate-limit retries"
        if kind is ErrorKind.SERVICE_UNAVAILABLE:
            return f"{route.label} exhausted service-unavailable retries"
        return f"{route.label} exhausted transient-error retries"

    @staticmethod
    def _unavailable_error(
        reason: str | None,
        error: Exception | None,
    ) -> ModelGatewayUnavailableError:
        unavailable = ModelGatewayUnavailableError(
            reason or "all configured model routes are unavailable"
        )
        if error is not None:
            unavailable.__cause__ = error
        return unavailable

__all__ = [
    "ModelGateway",
    "ModelGatewayEvent",
    "ModelGatewayEventHandler",
    "ModelGatewayEventType",
    "ModelGatewayRequestRejectedError",
    "ModelGatewayUnavailableError",
    "ModelRoute",
]
