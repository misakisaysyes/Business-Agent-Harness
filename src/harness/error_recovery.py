"""错误分类和有边界的恢复策略。

Error classification and bounded recovery strategies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Final

from harness.messages import Message

PROMPT_TOO_LONG_MARKERS: Final[tuple[str, ...]] = (
    "prompt_too_long",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
)
OUTPUT_LIMIT_FINISH_REASONS: Final[frozenset[str]] = frozenset(
    {"length", "max_tokens", "max_output_tokens"}
)


class ErrorKind(StrEnum):
    """跨模型 Provider 和 Tool 的稳定错误分类。

    Stable error categories shared across model providers and tools.
    """

    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    PROMPT_TOO_LONG = "prompt_too_long"
    OUTPUT_LIMIT = "output_limit"
    REQUEST_REJECTED = "request_rejected"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ErrorRecoveryPolicy:
    """一次模型调用允许使用的有界输出恢复策略。

    Bounded output-recovery policy for one model invocation.
    """

    initial_max_output_tokens: int = 4_096
    max_output_tokens: int = 16_384
    output_token_multiplier: float = 2.0
    max_output_retries: int = 1
    tool_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.initial_max_output_tokens < 1:
            raise ValueError("initial max output tokens must be at least 1")
        if self.max_output_tokens < self.initial_max_output_tokens:
            raise ValueError("max output tokens must not be below the initial limit")
        if self.output_token_multiplier <= 1:
            raise ValueError("output token multiplier must be greater than 1")
        if not 0 <= self.max_output_retries <= 2:
            raise ValueError("max output retries must be between 0 and 2")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("tool timeout must be greater than 0")


class PromptTooLongRecoveryError(RuntimeError):
    """Reactive Compact 后 Prompt 仍超过模型上下文窗口。"""


class OutputTokenRecoveryError(RuntimeError):
    """提高输出上限后模型响应仍因长度被截断。"""


def classify_error(error: BaseException) -> ErrorKind:
    """把 Provider 异常链映射到稳定错误分类。

    Map a provider exception chain to a stable error category.
    """

    chain = error_chain(error)
    if any(isinstance(item, (asyncio.CancelledError, KeyboardInterrupt)) for item in chain):
        return ErrorKind.CANCELLED
    if any(isinstance(item, OutputTokenRecoveryError) for item in chain):
        return ErrorKind.OUTPUT_LIMIT
    if any(isinstance(item, PromptTooLongRecoveryError) for item in chain):
        return ErrorKind.PROMPT_TOO_LONG
    if any(_contains_prompt_too_long_marker(item) for item in chain):
        return ErrorKind.PROMPT_TOO_LONG

    statuses = tuple(
        status for item in chain if (status := status_code(item)) is not None
    )
    if 429 in statuses:
        return ErrorKind.RATE_LIMIT
    if any(status == 408 or 500 <= status <= 599 for status in statuses):
        return ErrorKind.SERVICE_UNAVAILABLE
    if any(400 <= status <= 499 for status in statuses):
        return ErrorKind.REQUEST_REJECTED

    for item in chain:
        if isinstance(item, (TimeoutError, ConnectionError)):
            return ErrorKind.TRANSIENT
        name = type(item).__name__.casefold()
        if any(marker in name for marker in ("timeout", "connection", "ratelimit")):
            return ErrorKind.TRANSIENT
    return ErrorKind.UNKNOWN


def is_retryable_model_error(error: BaseException) -> bool:
    """返回模型错误是否适合无副作用重试。"""

    return classify_error(error) in {
        ErrorKind.TRANSIENT,
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVICE_UNAVAILABLE,
    }


def is_prompt_too_long_error(error: BaseException) -> bool:
    """识别常见 Provider 的上下文窗口溢出错误。"""

    return classify_error(error) is ErrorKind.PROMPT_TOO_LONG


def is_output_truncated(message: Message) -> bool:
    """判断模型是否因为输出 Token 上限停止生成。"""

    finish_reason = message.provider_metadata.get("finish_reason")
    return (
        isinstance(finish_reason, str)
        and finish_reason.casefold() in OUTPUT_LIMIT_FINISH_REASONS
    )


def next_output_token_limit(current: int, policy: ErrorRecoveryPolicy) -> int:
    """按倍率提高输出上限，但永远不超过配置的硬上限。"""

    increased = max(current + 1, int(current * policy.output_token_multiplier))
    return min(policy.max_output_tokens, increased)


def retry_after_seconds(
    error: BaseException,
    now: datetime | None = None,
) -> float | None:
    """从异常链的 HTTP ``Retry-After`` Header 读取等待秒数。"""

    for item in error_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        value = headers.get("retry-after")
        if value is None:
            value = headers.get("Retry-After")
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
            except (TypeError, ValueError, OverflowError):
                continue
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            current = now or datetime.now(UTC)
            return max(0.0, (retry_at - current).total_seconds())
    return None


def safe_error_reason(error: BaseException) -> str:
    """返回不包含请求正文和凭据的简短错误原因。"""

    status = next(
        (value for item in error_chain(error) if (value := status_code(item)) is not None),
        None,
    )
    if status is not None:
        return f"HTTP {status} {type(error).__name__}"
    return type(error).__name__


def status_code(error: BaseException) -> int | None:
    """读取常见 SDK 异常携带的 HTTP 状态码。"""

    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def has_status_code(error: BaseException, expected: int) -> bool:
    """检查异常链中是否包含指定 HTTP 状态码。"""

    return any(status_code(item) == expected for item in error_chain(error))


def error_chain(error: BaseException) -> tuple[BaseException, ...]:
    """返回去重后的异常 cause/context 链。"""

    chain: list[BaseException] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _contains_prompt_too_long_marker(error: BaseException) -> bool:
    text = str(error).casefold()
    return any(marker in text for marker in PROMPT_TOO_LONG_MARKERS)


__all__ = [
    "ErrorKind",
    "ErrorRecoveryPolicy",
    "OutputTokenRecoveryError",
    "PromptTooLongRecoveryError",
    "classify_error",
    "error_chain",
    "has_status_code",
    "is_output_truncated",
    "is_prompt_too_long_error",
    "is_retryable_model_error",
    "next_output_token_limit",
    "retry_after_seconds",
    "safe_error_reason",
    "status_code",
]
