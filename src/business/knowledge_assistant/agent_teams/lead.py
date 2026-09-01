"""负责研究任务协调的 Lead Agent 角色。

Lead Agent role for research-task coordination.
"""

import json
from typing import Literal
from uuid import uuid4

from pydantic import JsonValue

from business.knowledge_assistant.agent_teams.analyst import ANALYST
from business.knowledge_assistant.agent_teams.researcher import (
    CATALOG_RESEARCHER,
    RAG_RESEARCHER,
    WEB_RESEARCHER,
)
from business.knowledge_assistant.agent_teams.reviewer import (
    REVIEWER,
    ReviewDecision,
    ReviewOutput,
)
from business.knowledge_assistant.search_routing import (
    SearchMode,
    classify_search_query,
)
from harness.capabilities.agent_teams.contracts import (
    SubagentContext,
    SubagentTask,
)
from harness.capabilities.agent_teams.team import TeamCoordinator
from harness.capabilities.agent_teams.team_protocols import TeamMessageKind
from harness.messages import ToolResult, ToolUse
from harness.tool_use import ToolExecutionContext, ToolInput


class DelegateResearchInput(ToolInput):
    """Lead 委派一次 Researcher 任务的参数。"""

    objective: str
    research_kind: Literal["auto", "catalog", "rag", "web"] = "auto"


class DelegateAnalysisInput(ToolInput):
    """Lead 委派分析任务的参数。"""

    objective: str
    evidence: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()


class RequestReviewInput(ToolInput):
    """Lead 请求 Reviewer 审核候选结果的参数。"""

    candidate: dict[str, JsonValue]
    evidence: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()


_KIND_TO_ROLE = {
    "catalog": CATALOG_RESEARCHER,
    "rag": RAG_RESEARCHER,
    "web": WEB_RESEARCHER,
}


class DelegateResearchTool:
    """把一次 Researcher 委派暴露为只读 Agent-as-a-Tool。"""

    name = "delegate_research"
    description = (
        "Delegate one focused evidence-retrieval task to an isolated Researcher. "
        "Use research_kind catalog/rag/web explicitly; for hybrid research call this "
        "tool separately for rag and web."
    )
    input_schema = DelegateResearchInput
    concurrency_group = None

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self.coordinator = coordinator

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "error": "missing_runtime_context",
                "message": "delegate_research requires trusted runtime context",
            },
            is_error=True,
        )

    async def ainvoke_with_context(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool_input = DelegateResearchInput.model_validate(tool_use.input)
        metadata = context.metadata
        raw_search_mode = metadata.get("search_mode", SearchMode.AUTO.value)
        try:
            search_mode = SearchMode(str(raw_search_mode))
        except ValueError:
            search_mode = SearchMode.AUTO

        kind = tool_input.research_kind
        if kind == "auto":
            if search_mode is SearchMode.AUTO:
                plan = classify_search_query(tool_input.objective)
                if plan.mode is SearchMode.HYBRID:
                    return self._error(
                        tool_use,
                        "hybrid research must be split into separate rag and web delegations",
                    )
                kind = plan.mode.value
            elif search_mode is SearchMode.CATALOG:
                kind = "catalog"
            elif search_mode is SearchMode.RAG:
                kind = "rag"
            elif search_mode is SearchMode.WEB:
                kind = "web"
            else:
                return self._error(
                    tool_use,
                    "hybrid research must be split into separate rag and web delegations",
                )

        if not self._kind_allowed(kind, search_mode):
            return self._error(
                tool_use,
                f"research kind {kind} is not allowed in search mode {search_mode.value}",
            )

        parent_thread_id = metadata.get("parent_thread_id", context.thread_id)
        if not isinstance(parent_thread_id, str) or not parent_thread_id:
            parent_thread_id = context.thread_id
        parent_run_id = metadata.get("run_id")
        user_id = metadata.get("user_id")
        team_run_id = metadata.get("team_run_id")
        if not isinstance(team_run_id, str) or not team_run_id:
            team_run_id = f"team_{uuid4().hex}"
        task_context: dict[str, JsonValue] = {
            "research_kind": kind,
            "requested_search_mode": search_mode.value,
        }
        subagent_context = SubagentContext(
            parent_thread_id=parent_thread_id,
            parent_run_id=parent_run_id if isinstance(parent_run_id, str) else None,
            team_run_id=team_run_id,
            user_id=user_id if isinstance(user_id, str) else None,
            search_mode=search_mode.value,
            allowed_tool_names=(),
            metadata={
                "user_id": user_id if isinstance(user_id, str) else None,
            },
        )
        result = await self.coordinator.delegate(
            SubagentTask(
                role=_KIND_TO_ROLE[kind],
                objective=tool_input.objective,
                context=task_context,
            ),
            subagent_context,
        )
        return ToolResult(
            tool_use_id=tool_use.id,
            content=result.model_dump(mode="json"),
            is_error=not result.succeeded,
        )

    @staticmethod
    def _kind_allowed(kind: str, mode: SearchMode) -> bool:
        if mode is SearchMode.RAG:
            return kind in {"catalog", "rag"}
        if mode is SearchMode.WEB:
            return kind == "web"
        if mode is SearchMode.CATALOG:
            return kind == "catalog"
        return kind in {"catalog", "rag", "web"}

    @staticmethod
    def _error(tool_use: ToolUse, message: str) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use.id,
            content={"error": "invalid_research_delegation", "message": message},
            is_error=True,
        )


class DelegateAnalysisTool:
    """把 Lead 收集到的证据交给隔离 Analyst。"""

    name = "delegate_analysis"
    description = (
        "Delegate supplied evidence to an isolated Analyst for comparison, calculation, "
        "and bounded conclusions. The Analyst cannot retrieve unrelated data."
    )
    input_schema = DelegateAnalysisInput
    concurrency_group = None

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self.coordinator = coordinator

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "error": "missing_runtime_context",
                "message": "delegate_analysis requires trusted runtime context",
            },
            is_error=True,
        )

    async def ainvoke_with_context(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool_input = DelegateAnalysisInput.model_validate(tool_use.input)
        subagent_context = _subagent_context(context)
        result = await self.coordinator.delegate(
            SubagentTask(
                role=ANALYST,
                objective=tool_input.objective,
                context={
                    "evidence": list(tool_input.evidence),
                    "citation_ids": list(tool_input.citation_ids),
                },
            ),
            subagent_context,
        )
        return ToolResult(
            tool_use_id=tool_use.id,
            content=result.model_dump(mode="json"),
            is_error=not result.succeeded,
        )


class RequestReviewTool:
    """把最终候选结果交给无写入权限的 Reviewer。"""

    name = "request_review"
    description = (
        "Ask an isolated Reviewer to check a candidate result, evidence, calculations, "
        "citations, and uncertainty boundaries before the Lead finalizes it."
    )
    input_schema = RequestReviewInput
    concurrency_group = None

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self.coordinator = coordinator

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use.id,
            content={
                "error": "missing_runtime_context",
                "message": "request_review requires trusted runtime context",
            },
            is_error=True,
        )

    async def ainvoke_with_context(
        self,
        tool_use: ToolUse,
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool_input = RequestReviewInput.model_validate(tool_use.input)
        subagent_context = _subagent_context(context)
        reservation = await self.coordinator.reserve_review_round(subagent_context)
        if reservation is None:
            return ToolResult(
                tool_use_id=tool_use.id,
                content={
                    "error": "review_limit_exceeded",
                    "message": "review loop exceeded the maximum of two revisions",
                },
                is_error=True,
            )
        team_run_id, review_round = reservation
        task = SubagentTask(
            role=REVIEWER,
            objective="Review the candidate result against the supplied evidence.",
            context={
                "candidate": tool_input.candidate,
                "evidence": list(tool_input.evidence),
                "citation_ids": list(tool_input.citation_ids),
            },
        )
        protocol_context = subagent_context.model_copy(
            update={"team_run_id": team_run_id}
        )
        await self.coordinator.emit_protocol(
            task,
            protocol_context,
            TeamMessageKind.REVIEW_REQUEST,
            {"review_round": review_round},
            sender="lead",
            recipient=REVIEWER,
        )
        result = await self.coordinator.delegate(
            task,
            protocol_context,
        )
        review = _parse_review_output(result.summary)
        if review is not None:
            await self.coordinator.emit_protocol(
                task,
                protocol_context,
                TeamMessageKind.REVIEW_RESULT,
                review.model_dump(mode="json"),
                sender=REVIEWER,
                recipient="lead",
            )
            if review.decision is ReviewDecision.NEEDS_REVISION:
                await self.coordinator.emit_protocol(
                    task,
                    protocol_context,
                    TeamMessageKind.REVISION_REQUEST,
                    review.model_dump(mode="json"),
                    sender=REVIEWER,
                    recipient="lead",
                )
        content = result.model_dump(mode="json")
        content["review_round"] = review_round
        if review is not None:
            content["review"] = review.model_dump(mode="json")
        return ToolResult(
            tool_use_id=tool_use.id,
            content=content,
            is_error=not result.succeeded,
        )


def _parse_review_output(summary: str) -> ReviewOutput | None:
    """解析 Reviewer 要求的 JSON；解析失败时保留原始摘要给 Lead。"""

    candidate = summary.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").removeprefix("json").strip()
    try:
        return ReviewOutput.model_validate(json.loads(candidate))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _subagent_context(context: ToolExecutionContext) -> SubagentContext:
    """从可信 Tool 上下文创建不透传主会话消息的子 Agent 上下文。"""

    metadata = context.metadata
    raw_mode = metadata.get("search_mode", SearchMode.AUTO.value)
    search_mode = str(raw_mode)
    team_run_id = metadata.get("team_run_id")
    parent_run_id = metadata.get("run_id")
    user_id = metadata.get("user_id")
    parent_thread_id = metadata.get("parent_thread_id")
    if not isinstance(parent_thread_id, str) or not parent_thread_id:
        parent_thread_id = context.thread_id
    return SubagentContext(
        parent_thread_id=parent_thread_id,
        parent_run_id=parent_run_id if isinstance(parent_run_id, str) else None,
        team_run_id=team_run_id if isinstance(team_run_id, str) else None,
        user_id=user_id if isinstance(user_id, str) else None,
        search_mode=search_mode,
        metadata={"user_id": user_id if isinstance(user_id, str) else None},
    )


__all__ = [
    "DelegateAnalysisInput",
    "DelegateAnalysisTool",
    "DelegateResearchInput",
    "DelegateResearchTool",
    "RequestReviewInput",
    "RequestReviewTool",
]
