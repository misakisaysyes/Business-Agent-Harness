"""跨测试复用的轻量测试替身。

Lightweight test doubles shared across test suites.
"""

from harness.messages import Message
from harness.model import ModelRequest


class FakeModel:
    """记录请求并返回固定 Message 的 ModelProvider 测试替身。

    ModelProvider test double that records requests and returns a fixed message.
    """

    name = "fake"

    def __init__(self, response: Message) -> None:
        self.response = response
        self.sync_requests: list[ModelRequest] = []
        self.async_requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> Message:
        """记录同步请求并返回固定响应。

        Record a synchronous request and return the fixed response.
        """

        self.sync_requests.append(request)
        return self.response

    async def ainvoke(self, request: ModelRequest) -> Message:
        """记录异步请求并返回固定响应。

        Record an asynchronous request and return the fixed response.
        """

        self.async_requests.append(request)
        return self.response


class FakeSequenceModel:
    """按顺序返回多条 Message 的 ModelProvider 测试替身。

    ModelProvider test double returning a sequence of messages in order.
    """

    name = "fake_sequence"

    def __init__(self, responses: list[Message]) -> None:
        if not responses:
            raise ValueError("responses must not be empty")
        self.responses = responses
        self.sync_requests: list[ModelRequest] = []
        self.async_requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> Message:
        """记录请求并返回下一条同步响应。

        Record the request and return the next synchronous response.
        """

        response_index = len(self.sync_requests)
        self.sync_requests.append(request)
        if response_index >= len(self.responses):
            raise AssertionError("fake model response sequence exhausted")
        return self.responses[response_index]

    async def ainvoke(self, request: ModelRequest) -> Message:
        """记录请求并返回下一条异步响应。

        Record the request and return the next asynchronous response.
        """

        response_index = len(self.async_requests)
        self.async_requests.append(request)
        if response_index >= len(self.responses):
            raise AssertionError("fake model response sequence exhausted")
        return self.responses[response_index]
