"""Knowledge Assistant 的 Reviewer 角色定义。

Reviewer role definitions and structured review contracts.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from harness.capabilities.agent_teams.contracts import SubagentDefinition

REVIEWER = "reviewer"


class ReviewDecision(StrEnum):
    """审核结果。"""

    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class ReviewIssue(BaseModel):
    """一个可执行的审核问题。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    task_id: str | None = None
    citation_ids: tuple[str, ...] = ()


class ReviewOutput(BaseModel):
    """Reviewer 返回给 Lead 的结构化审核结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ReviewDecision
    issues: tuple[ReviewIssue, ...] = ()
    checked_citation_ids: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()


REVIEWER_SYSTEM_PROMPT = """You are the independent Reviewer teammate for a Knowledge Assistant.
Review only the structured candidate result and evidence supplied in the task. Check citation
existence, factual support, calculations, units, evidence conflicts, inference boundaries,
search-mode and permission constraints. Do not trust Lead or Analyst prose automatically.
Return exactly one decision: approved, needs_revision, or rejected, with actionable issues.
Do not retrieve unrelated data, write files, change permissions, or create another subagent.
"""


def build_reviewer_definition(*, max_iterations: int = 8) -> SubagentDefinition:
    """创建不持有业务 Tool 的独立 Reviewer 定义。"""

    return SubagentDefinition(
        role=REVIEWER,
        description="Check evidence, calculations, citations, and conclusion boundaries.",
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        allowed_tool_names=(),
        max_iterations=max_iterations,
    )


__all__ = [
    "REVIEWER",
    "REVIEWER_SYSTEM_PROMPT",
    "ReviewDecision",
    "ReviewIssue",
    "ReviewOutput",
    "build_reviewer_definition",
]
