"""M4-6 错误分类和恢复策略测试。

M4-6 error classification and recovery-policy tests.
"""

from harness.error_recovery import ErrorKind, classify_error, is_retryable_model_error


class StatusError(RuntimeError):
    """携带 HTTP 状态码的测试异常。"""

    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_model_errors_are_classified_without_provider_specific_types() -> None:
    """通用分类不得依赖某一个模型 SDK 的异常类。"""

    assert classify_error(TimeoutError("timeout")) is ErrorKind.TRANSIENT
    assert classify_error(StatusError(429)) is ErrorKind.RATE_LIMIT
    assert classify_error(StatusError(503)) is ErrorKind.SERVICE_UNAVAILABLE
    assert classify_error(StatusError(400)) is ErrorKind.REQUEST_REJECTED
    assert (
        classify_error(StatusError(400, "context_length_exceeded"))
        is ErrorKind.PROMPT_TOO_LONG
    )


def test_only_transient_rate_limit_and_service_errors_are_blindly_retried() -> None:
    """参数错误和 Prompt 过长必须交给专用恢复策略处理。"""

    assert is_retryable_model_error(TimeoutError("timeout"))
    assert is_retryable_model_error(StatusError(429))
    assert is_retryable_model_error(StatusError(503))
    assert not is_retryable_model_error(StatusError(400))
    assert not is_retryable_model_error(StatusError(400, "maximum context length"))
