"""Knowledge Assistant 的 Analyst 角色定义。

Analyst role definitions and structured output contracts.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from harness.capabilities.agent_teams.contracts import SubagentDefinition

ANALYST = "analyst"


class AnalystFinding(BaseModel):
    """一个由证据支持的分析发现。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = ()
    is_inference: bool = False


class AnalystCalculation(BaseModel):
    """分析过程中的可复核计算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expression: str = Field(min_length=1)
    result: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = ()


class AnalystOutput(BaseModel):
    """Analyst 返回给 Lead 的结构化结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[AnalystFinding, ...] = ()
    calculations: tuple[AnalystCalculation, ...] = ()
    conclusions: tuple[str, ...] = ()
    supporting_citations: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()


ANALYST_SYSTEM_PROMPT = """You are the Analyst teammate for a Knowledge Assistant.
Only analyze the evidence explicitly supplied in the task. Do not read the parent conversation
or retrieve unrelated data. Use Calculator for material arithmetic. Keep facts, calculations,
and inferences separate; preserve only citation IDs present in the supplied evidence. Return
findings, calculations, conclusions, supporting citations, and uncertainties. Do not write
files, change permissions, or create another subagent.
"""


def build_analyst_definition(
    available_tool_names: Iterable[str],
    *,
    max_iterations: int = 8,
) -> SubagentDefinition:
    """根据当前 Runtime 的 Tool 发现结果创建 Analyst 定义。"""

    available = frozenset(available_tool_names)
    calculator = ("calculator",) if "calculator" in available else ()
    return SubagentDefinition(
        role=ANALYST,
        description="Compare supplied evidence, calculate results, and form bounded conclusions.",
        system_prompt=ANALYST_SYSTEM_PROMPT,
        allowed_tool_names=calculator,
        max_iterations=max_iterations,
    )


__all__ = [
    "ANALYST",
    "ANALYST_SYSTEM_PROMPT",
    "AnalystCalculation",
    "AnalystFinding",
    "AnalystOutput",
    "build_analyst_definition",
]
