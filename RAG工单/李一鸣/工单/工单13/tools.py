"""Tool base class and calculator tool for ReAct agents."""

from __future__ import annotations

import ast
import operator


class BaseTool:
    name = "base"
    description = "base tool"

    def execute(self, argument: str) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Calculate a basic arithmetic expression."

    OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def execute(self, argument: str) -> str:
        return str(self._eval(ast.parse(argument, mode="eval").body))

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPERATORS:
            return self.OPERATORS[type(node.op)](self._eval(node.left), self._eval(node.right))
        raise ValueError("unsupported expression")


class ToolRegistry:
    def __init__(self, tools=None):
        self.tools = {tool.name: tool for tool in (tools or [CalculatorTool()])}

    def execute(self, name: str, argument: str) -> str:
        if name not in self.tools:
            raise KeyError(name)
        return self.tools[name].execute(argument)
