"""显式启用的真实模型冒烟测试。

Explicitly enabled live-model smoke test.
"""

import pytest

from entrypoints.bootstrap import bootstrap_agent
from harness.messages import Message, MessageRole
from harness.state import AgentState
from services.config import get_settings


@pytest.mark.live_model
def test_knowledge_assistant_with_live_model() -> None:
    """配置真实凭据后验证 Knowledge Assistant 普通问答。

    Validate Knowledge Assistant Q&A when real credentials are configured.
    """

    settings = get_settings()
    model = settings.model

    if model.provider != "anthropic" or model.model_id is None or model.api_key is None:
        pytest.skip("live Anthropic model configuration is not available")

    if model.api_key.get_secret_value().startswith("replace-with-"):
        pytest.skip("placeholder Anthropic API key is not a live credential")

    loop = bootstrap_agent(settings=settings)
    state: AgentState = {
        "thread_id": "live-model-smoke-test",
        "messages": [Message(role=MessageRole.USER, content="请用一句话介绍你能做什么")],
    }

    result = loop.invoke(state)

    assert result["messages"][-1].role is MessageRole.ASSISTANT
    assert result["messages"][-1].content
