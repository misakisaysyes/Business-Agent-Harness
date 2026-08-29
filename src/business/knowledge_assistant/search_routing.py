"""Knowledge Assistant 的 RAG/目录/联网搜索路由策略。"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from harness.context import ContextFragment
from harness.messages import MessageRole
from harness.state import AgentState


class SearchMode(StrEnum):
    """一次查询需要使用的检索来源。"""

    AUTO = "auto"
    CATALOG = "catalog"
    RAG = "rag"
    WEB = "web"
    HYBRID = "hybrid"


class SearchPlan(BaseModel):
    """不包含用户原文的查询路由计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: SearchMode
    use_catalog: bool
    use_rag: bool
    use_web: bool


def forced_search_plan(mode: SearchMode) -> SearchPlan:
    """将 CLI/API 指定的模式转换为不可歧义的计划。"""

    if mode is SearchMode.AUTO:
        raise ValueError("auto mode must be resolved from the user query")
    return SearchPlan(
        mode=mode,
        use_catalog=mode is SearchMode.CATALOG,
        use_rag=mode in {SearchMode.RAG, SearchMode.CATALOG, SearchMode.HYBRID},
        use_web=mode in {SearchMode.WEB, SearchMode.HYBRID},
    )


def is_web_search_tool_name(name: str) -> bool:
    """识别约定命名的联网搜索 Tool，包括 MCP 命名空间。"""

    normalized = name.casefold().replace("-", "_")
    return any(
        marker in normalized
        for marker in ("web_search", "internet_search", "browser_search", "search_web")
    )


_CATALOG_TERMS = (
    "多少",
    "几次",
    "几份",
    "几场",
    "总数",
    "列出",
    "有哪些",
    "全部",
    "统计",
    "枚举",
)
_WEB_TERMS = (
    "最新",
    "今天",
    "当前",
    "现在",
    "近期",
    "新闻",
    "价格",
    "行情",
    "联网",
    "网上",
    "互联网",
)
_INTERNAL_TERMS = (
    "我",
    "我的",
    "公司",
    "内部",
    "知识库",
    "文档",
    "面试",
    "项目",
    "政策",
    "记录",
)


def classify_search_query(query: str) -> SearchPlan:
    """按稳定规则判断查询来源；模型负责后续执行和综合。"""

    normalized = query.casefold()
    has_catalog_intent = any(term in normalized for term in _CATALOG_TERMS)
    has_web_intent = any(term in normalized for term in _WEB_TERMS)
    has_internal_intent = any(term in normalized for term in _INTERNAL_TERMS)
    use_catalog = has_catalog_intent and has_internal_intent
    use_web = has_web_intent
    use_rag = has_internal_intent or not use_web

    if use_catalog and use_web:
        mode = SearchMode.HYBRID
    elif use_catalog:
        mode = SearchMode.CATALOG
    elif use_web and use_rag:
        mode = SearchMode.HYBRID
    elif use_web:
        mode = SearchMode.WEB
    else:
        mode = SearchMode.RAG
    return SearchPlan(
        mode=mode,
        use_catalog=use_catalog,
        use_rag=use_rag,
        use_web=use_web,
    )


class SearchRoutingContextProvider:
    """在模型请求前注入当前查询的最小路由指导，不注入用户原文。"""

    name = "search_routing"

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        user_message = next(
            (
                message.content
                for message in reversed(state["messages"])
                if message.role is MessageRole.USER
            ),
            "",
        )
        if not user_message.strip():
            return ()
        raw_mode = state.get("metadata", {}).get("search_mode", SearchMode.AUTO.value)
        try:
            selected_mode = SearchMode(str(raw_mode))
        except ValueError:
            selected_mode = SearchMode.AUTO
        plan = (
            classify_search_query(user_message)
            if selected_mode is SearchMode.AUTO
            else forced_search_plan(selected_mode)
        )
        instructions = {
            SearchMode.CATALOG: (
                "必须先调用 document_catalog 统计或枚举文档；"
                "不要从 document_search 的 Top-K 结果计数。"
            ),
            SearchMode.RAG: "调用 document_search 查询授权内部知识；不要无依据联网。",
            SearchMode.WEB: (
                "调用已提供的联网搜索工具查询最新公共信息；不要把私有文档原文发送到联网工具。"
            ),
            SearchMode.HYBRID: (
                "同时查询授权内部知识和最新公共信息；私有查询先脱敏，分别保留两类来源并说明冲突。"
            ),
            SearchMode.AUTO: "根据问题内容选择合适的检索来源。",
        }[plan.mode]
        return (
            ContextFragment(
                key="search-routing-current",
                title="Search Routing Guidance",
                content=(
                    f"Recommended mode: {plan.mode.value}. {instructions} "
                    "如果所需联网工具未提供，明确说明无法完成联网部分。"
                ),
                priority=950,
            ),
        )


__all__ = [
    "SearchMode",
    "SearchPlan",
    "SearchRoutingContextProvider",
    "classify_search_query",
    "forced_search_plan",
    "is_web_search_tool_name",
]
