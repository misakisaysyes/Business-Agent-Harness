"""模型 Token 用量的请求级收集和用户级账本。

Request-scoped model-token collection and user-scoped accounting.
"""

import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.messages import Message
from harness.model import ModelProvider, ModelRequest


class TokenUsage(BaseModel):
    """一次或多次模型响应的标准 Token 用量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @classmethod
    def combine(cls, usages: Sequence["TokenUsage"]) -> "TokenUsage":
        """累加多次真实模型响应中的用量。"""

        return cls(
            input_tokens=sum(usage.input_tokens for usage in usages),
            output_tokens=sum(usage.output_tokens for usage in usages),
            total_tokens=sum(usage.total_tokens for usage in usages),
        )

    @classmethod
    def from_message(cls, message: Message) -> "TokenUsage | None":
        """从 Provider 标准化元数据读取用量；缺失时不估算。"""

        raw_usage = message.provider_metadata.get("usage")
        if not isinstance(raw_usage, dict):
            return None
        usage = cast(dict[str, object], raw_usage)
        values = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(value := usage.get(key), int) and not isinstance(value, bool)
        }
        if not values:
            return None
        try:
            return cls.model_validate(values)
        except ValidationError:
            return None


class ModelUsageCollector:
    """使用 ContextVar 隔离并发请求中的模型用量。"""

    def __init__(self) -> None:
        self._usages: ContextVar[list[TokenUsage] | None] = ContextVar(
            "model_token_usages",
            default=None,
        )

    def emit(self, message: Message) -> None:
        """记录当前请求的一条模型响应用量。"""

        usages = self._usages.get()
        usage = TokenUsage.from_message(message)
        if usages is not None and usage is not None:
            usages.append(usage)

    @contextmanager
    def capture(self) -> Generator[list[TokenUsage]]:
        """创建一个与其他并发请求隔离的用量收集上下文。"""

        usages: list[TokenUsage] = []
        token = self._usages.set(usages)
        try:
            yield usages
        finally:
            self._usages.reset(token)


class UsageTrackingModel:
    """为共享 ModelProvider 增加无侵入的请求级用量收集。"""

    def __init__(self, model: ModelProvider, collector: ModelUsageCollector) -> None:
        self.model = model
        self.collector = collector

    @property
    def name(self) -> str:
        return self.model.name

    def invoke(self, request: ModelRequest) -> Message:
        response = self.model.invoke(request)
        self.collector.emit(response)
        return response

    async def ainvoke(self, request: ModelRequest) -> Message:
        response = await self.model.ainvoke(request)
        self.collector.emit(response)
        return response


class UserTokenUsageLedger:
    """保存当前进程中相互隔离的用户 Token 总量。"""

    def __init__(self) -> None:
        self._totals: dict[str, TokenUsage] = {}
        self._lock = threading.Lock()

    def record(self, user_id: str, usages: Sequence[TokenUsage]) -> TokenUsage:
        """把当前请求的真实用量累加到指定用户。"""

        request_usage = TokenUsage.combine(usages)
        with self._lock:
            current = self._totals.get(user_id, TokenUsage())
            total = TokenUsage.combine((current, request_usage))
            self._totals[user_id] = total
        return request_usage

    def get(self, user_id: str) -> TokenUsage:
        """返回指定用户的累计用量。"""

        with self._lock:
            return self._totals.get(user_id, TokenUsage())


__all__ = [
    "ModelUsageCollector",
    "TokenUsage",
    "UsageTrackingModel",
    "UserTokenUsageLedger",
]
