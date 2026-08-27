"""受限的算术计算 Tool。

Restricted arithmetic tool.
"""

import ast
import math

from pydantic import Field

from harness.messages import ToolResult, ToolUse
from harness.tool_use import ToolInput

MAX_EXPRESSION_NODES = 50
MAX_ABSOLUTE_RESULT = 1_000_000_000_000_000
MAX_EXPONENT = 10


class CalculatorInput(ToolInput):
    """Calculator Tool 的参数。

    Input for the calculator tool.
    """

    expression: str = Field(min_length=1, max_length=200)


def _ensure_safe_number(value: object) -> int | float:
    """拒绝非有限值和过大的计算结果。

    Reject non-finite and excessively large calculation results.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("calculation must produce a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("calculation produced a non-finite result")
    if abs(value) > MAX_ABSOLUTE_RESULT:
        raise ValueError("calculation result exceeds the allowed range")
    return value


def _evaluate(node: ast.AST) -> int | float:
    """递归计算白名单 AST 节点。

    Recursively evaluate allowlisted AST nodes.
    """

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only integer and floating-point literals are allowed")
        return _ensure_safe_number(node.value)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _ensure_safe_number(-operand)
        raise ValueError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("exponent exceeds the allowed range")
            result = left**right
        else:
            raise ValueError("unsupported binary operator")

        return _ensure_safe_number(result)

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> int | float:
    """解析并计算一个受限算术表达式。

    Parse and evaluate one restricted arithmetic expression.
    """

    parsed = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(parsed)) > MAX_EXPRESSION_NODES:
        raise ValueError("expression is too complex")
    return _evaluate(parsed)


class CalculatorTool:
    """不使用 eval 的受限算术 Tool。

    Restricted arithmetic tool that does not use eval.
    """

    name = "calculator"
    description = "Calculate a restricted arithmetic expression using numbers and operators."
    input_schema = CalculatorInput
    concurrency_group = None

    async def ainvoke(self, tool_use: ToolUse) -> ToolResult:
        """计算表达式并返回与 ToolUse 配对的结果。

        Calculate the expression and return a result paired with the tool use.
        """

        tool_input = CalculatorInput.model_validate(tool_use.input)
        return ToolResult(
            tool_use_id=tool_use.id,
            content=calculate(tool_input.expression),
        )


__all__ = ["CalculatorInput", "CalculatorTool", "calculate"]
