"""Minimal Thought -> Action -> Observation ReAct agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import config


@dataclass
class Tool:
    name: str
    description: str
    execute: Callable[[str], str]


class ReActAgent:
    def __init__(self, llm, tools: list[Tool], settings=config):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.settings = settings

    def reason(self, question: str, trace: list[str]) -> str:
        tool_list = "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())
        prompt = f"Question: {question}\nTools:\n{tool_list}\nTrace:\n" + "\n".join(trace)
        return self.llm.invoke(prompt)

    def act(self, action: str) -> str:
        name, _, argument = action.partition(":")
        tool = self.tools.get(name.strip())
        if not tool:
            return f"Unknown tool: {name.strip()}"
        try:
            return tool.execute(argument.strip())
        except Exception as exc:
            return f"Tool error: {exc}"

    def run(self, question: str) -> str:
        trace = []
        for _ in range(self.settings.max_iterations):
            thought = self.reason(question, trace)
            if thought.startswith("Final:"):
                return thought.removeprefix("Final:").strip()
            observation = self.act(thought.removeprefix("Action:").strip())
            trace.extend([f"Thought: {thought}", f"Observation: {observation}"])
        return "达到最大推理轮次，无法给出稳定答案。"
