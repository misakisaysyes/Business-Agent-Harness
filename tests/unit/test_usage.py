"""模型 Token 用量收集和用户账本测试。"""

import asyncio

from harness.messages import Message, MessageRole
from services.usage import ModelUsageCollector, TokenUsage, UserTokenUsageLedger


async def test_concurrent_usage_capture_is_request_scoped() -> None:
    """并发请求的用量不得进入对方的 ContextVar 收集器。"""

    collector = ModelUsageCollector()
    ready = asyncio.Event()
    active = 0

    async def capture(total: int) -> list[TokenUsage]:
        nonlocal active
        with collector.capture() as usages:
            active += 1
            if active == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            collector.emit(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="done",
                    provider_metadata={
                        "usage": {
                            "input_tokens": total - 1,
                            "output_tokens": 1,
                            "total_tokens": total,
                        }
                    },
                )
            )
            return usages

    first, second = await asyncio.gather(capture(10), capture(20))

    assert [usage.total_tokens for usage in first] == [10]
    assert [usage.total_tokens for usage in second] == [20]


def test_user_usage_ledger_keeps_totals_separate() -> None:
    """用户账本不得合并不同用户的模型用量。"""

    ledger = UserTokenUsageLedger()
    ledger.record("alice", (TokenUsage(input_tokens=8, output_tokens=2, total_tokens=10),))
    ledger.record("alice", (TokenUsage(input_tokens=4, output_tokens=1, total_tokens=5),))
    ledger.record("bob", (TokenUsage(input_tokens=16, output_tokens=4, total_tokens=20),))

    assert ledger.get("alice").total_tokens == 15
    assert ledger.get("bob").total_tokens == 20
