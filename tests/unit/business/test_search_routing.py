"""RAG、目录和联网搜索路由策略测试。"""

from business.knowledge_assistant.search_routing import (
    SearchMode,
    SearchRoutingContextProvider,
    classify_search_query,
)
from harness.messages import Message, MessageRole


def test_classify_internal_count_as_catalog() -> None:
    plan = classify_search_query("我面了字节几次？")

    assert plan.mode is SearchMode.CATALOG
    assert plan.use_catalog
    assert plan.use_rag
    assert not plan.use_web


def test_classify_current_internal_question_as_hybrid() -> None:
    plan = classify_search_query("公司的退款政策符合当前法律吗？")

    assert plan.mode is SearchMode.HYBRID
    assert plan.use_rag
    assert plan.use_web


def test_classify_current_public_question_as_web() -> None:
    plan = classify_search_query("当前机器人行业有哪些新闻？")

    assert plan.mode is SearchMode.WEB
    assert plan.use_web
    assert not plan.use_rag


def test_routing_context_does_not_echo_user_query() -> None:
    query = "我的私密项目名称不要出现在路由上下文里"
    provider = SearchRoutingContextProvider()
    fragments = provider.provide(
        {
            "thread_id": "thread",
            "messages": [Message(role=MessageRole.USER, content=query)],
        }
    )

    assert fragments
    assert query not in fragments[0].content
