"""MessageBus 协议和进程内实现。

MessageBus protocol and in-process implementation.
"""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from inspect import isawaitable
from typing import Protocol

from harness.capabilities.agent_teams.team_protocols import TeamMessage

MessageHandler = Callable[[TeamMessage], Awaitable[None] | None]


class MessageBusError(RuntimeError):
    """MessageBus 的基础错误。"""


class MessageBusClosedError(MessageBusError):
    """Team 已进入关闭流程，不能再投递消息。"""


class MessageBus(Protocol):
    """可替换为第四阶段持久化实现的最小总线接口。"""

    async def send(self, message: TeamMessage) -> bool:
        """向指定 recipient 投递一条消息；重复消息返回 False。"""
        ...

    async def publish(self, topic: str, message: TeamMessage) -> bool:
        """向 topic 的订阅者发布一条事件。"""
        ...

    def subscribe(self, topic: str, handler: MessageHandler) -> Callable[[], None]:
        """订阅 topic，并返回取消订阅函数。"""
        ...

    async def close(self, team_run_id: str) -> None:
        """拒绝新消息并等待当前消息完成。"""
        ...

    async def drain(self, team_run_id: str) -> None:
        """等待指定 Team 已经开始处理的消息完成。"""
        ...


class InMemoryMessageBus:
    """进程内、按 Team 隔离的异步 MessageBus。

    Messages are delivered in subscription order for each topic. A message is
    deduplicated by ``team_run_id + message_id`` so retrying delivery does not
    invoke a handler twice in the same Team run.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._topic_locks: dict[str, asyncio.Lock] = {}
        self._closed: set[str] = set()
        self._seen: set[tuple[str, str]] = set()
        self._messages: dict[str, list[TeamMessage]] = defaultdict(list)
        self._active: dict[str, set[asyncio.Task[object]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    @property
    def closed_team_runs(self) -> frozenset[str]:
        """返回已关闭 Team 的只读快照。"""

        return frozenset(self._closed)

    def messages(self, team_run_id: str) -> tuple[TeamMessage, ...]:
        """返回指定 Team 已接受的消息，便于测试和审计。"""

        return tuple(self._messages.get(team_run_id, ()))

    async def send(self, message: TeamMessage) -> bool:
        """向 recipient 投递消息。"""

        return await self._deliver(message, (message.recipient,))

    async def publish(self, topic: str, message: TeamMessage) -> bool:
        """向 topic 发布事件。"""

        if not topic.strip():
            raise ValueError("topic must not be blank")
        return await self._deliver(message, (topic,))

    def subscribe(self, topic: str, handler: MessageHandler) -> Callable[[], None]:
        """注册订阅者；取消函数可重复调用。"""

        if not topic.strip():
            raise ValueError("topic must not be blank")
        handlers = self._handlers[topic]
        handlers.append(handler)
        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            with suppress(ValueError):
                handlers.remove(handler)

        return unsubscribe

    async def close(self, team_run_id: str) -> None:
        """标记 Team 关闭并等待当前处理任务。"""

        if not team_run_id.strip():
            raise ValueError("team_run_id must not be blank")
        self._closed.add(team_run_id)
        await self.drain(team_run_id)

    async def drain(self, team_run_id: str) -> None:
        """等待已开始的消息处理，不接受新任务。"""

        while True:
            active = tuple(self._active.get(team_run_id, ()))
            if not active:
                return
            await asyncio.gather(*active)

    async def _deliver(self, message: TeamMessage, topics: tuple[str, ...]) -> bool:
        async with self._lock:
            if message.team_run_id in self._closed:
                raise MessageBusClosedError(
                    f"team run is closed: {message.team_run_id}"
                )
            dedupe_key = (message.team_run_id, message.message_id)
            if dedupe_key in self._seen:
                return False
            self._seen.add(dedupe_key)
            self._messages[message.team_run_id].append(message)

        current = asyncio.current_task()
        if current is not None:
            self._active[message.team_run_id].add(current)
        try:
            for topic in topics:
                handlers = tuple(self._handlers.get(topic, ()))
                if not handlers:
                    continue
                topic_lock = self._topic_locks.setdefault(topic, asyncio.Lock())
                async with topic_lock:
                    for handler in handlers:
                        result = handler(message)
                        if isawaitable(result):
                            await result
            return True
        finally:
            if current is not None:
                active = self._active.get(message.team_run_id)
                if active is not None:
                    active.discard(current)
                    if not active:
                        self._active.pop(message.team_run_id, None)


__all__ = [
    "InMemoryMessageBus",
    "MessageBus",
    "MessageBusClosedError",
    "MessageBusError",
    "MessageHandler",
]
