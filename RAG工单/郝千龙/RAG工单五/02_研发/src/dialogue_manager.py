# -*- coding: utf-8 -*-
# 【多轮对话管理模块 · dialogue_manager.py】对话历史滑窗、实体与意图状态维护
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""多轮对话状态管理层：

维护一个会话的对话历史、话题实体栈、上一轮意图，并提供：
- 滑窗截断（按轮数与字符数双重限制）；
- 实体栈回溯（供指代消解寻找先行词）；
- 改写查询入口（串联 query_understanding）。
"""
from dataclasses import dataclass, field
from typing import List, Optional

from config import CONFIG
from query_understanding import ParsedQuery, QueryUnderstanding


@dataclass
class Turn:
    """单轮对话记录。"""

    turn_no: int
    user_query: str
    rewritten_query: str
    entity: str
    intent: str
    answer: str = ""
    latency_s: float = 0.0


@dataclass
class DialogueState:
    """对话状态：历史轮次 + 实体栈 + 上一轮意图。"""

    turns: List[Turn] = field(default_factory=list)
    entity_stack: List[str] = field(default_factory=list)   # 最新在前

    @property
    def last_intent(self) -> str:
        """上一轮的问题意图。"""
        return self.turns[-1].intent if self.turns else ""

    @property
    def last_entity(self) -> str:
        """上一轮的话题实体。"""
        return self.turns[-1].entity if self.turns else ""

    def push_entity(self, entity: str) -> None:
        """把新实体压入栈顶（去重后置于最前）。"""
        if not entity:
            return
        if entity in self.entity_stack:
            self.entity_stack.remove(entity)
        self.entity_stack.insert(0, entity)
        # 滑窗：只保留最近若干实体
        self.entity_stack = self.entity_stack[: CONFIG.entity_window_turns]


class DialogueManager:
    """多轮对话管理器：串联 Query 理解与状态维护。"""

    def __init__(self) -> None:
        self.understanding = QueryUnderstanding()

    def process_turn(self, state: DialogueState, user_query: str) -> ParsedQuery:
        """处理一轮用户输入，返回改写后的查询并更新状态。

        :param state: 当前对话状态
        :param user_query: 用户原始问题
        :return: 解析结果（含改写查询、实体、意图）
        """
        parsed = self.understanding.rewrite(
            user_query,
            history_entities=state.entity_stack,
            last_intent=state.last_intent,
            last_entity=state.last_entity,
        )
        # 更新实体栈
        if parsed.entity:
            state.push_entity(parsed.entity)
        return parsed

    def record_answer(self, state: DialogueState, parsed: ParsedQuery,
                      answer: str, latency_s: float) -> None:
        """记录一轮问答结果到历史，并执行滑窗截断。

        :param state: 对话状态
        :param parsed: 本轮解析结果
        :param answer: 系统答案
        :param latency_s: 端到端耗时
        """
        turn = Turn(
            turn_no=len(state.turns) + 1,
            user_query=parsed.original,
            rewritten_query=parsed.rewritten,
            entity=parsed.entity,
            intent=parsed.intent,
            answer=answer,
            latency_s=latency_s,
        )
        state.turns.append(turn)
        # 滑窗：按轮数截断
        if len(state.turns) > CONFIG.max_history_turns:
            state.turns = state.turns[-CONFIG.max_history_turns:]
