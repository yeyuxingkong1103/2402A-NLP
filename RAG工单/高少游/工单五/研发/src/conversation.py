# -*- coding: utf-8 -*-
"""会话状态管理（本工单核心新增）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

职责：维护多轮对话的「记忆」，为 Query 改写提供上下文：
    - 最近轮次的话题实体（哪家公司、哪个工程/标准）
    - 最近一轮的答案类型与意图骨架（用于「那 X 呢？」式省略补全）
    - 当前活跃文档（用于文档级路由的继承）
    - 历史问答文本（用于界面展示与可选 LLM 辅助改写）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src import config


@dataclass
class Turn:
    """一轮问答记录。"""
    index: int
    question: str                      # 用户原始问句
    rewritten: str = ""                # 改写后的独立问句
    answer: str = ""                   # 系统答案
    doc: str = ""                      # 命中的源文档
    answer_type: str = ""              # 答案类型
    entities: List[str] = field(default_factory=list)   # 本轮实体
    keywords: List[str] = field(default_factory=list)   # 本轮关键词
    intent: str = ""                   # 意图骨架（剥离实体后的问句核心）
    evidence: List[dict] = field(default_factory=list)  # 证据片段（溯源）


class Conversation:
    """一次多轮对话会话。"""

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or config.MAX_TURNS
        self.turns: List[Turn] = []

    # -- 读写 ----------------------------------------------------------------
    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    @property
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def reset(self) -> None:
        self.turns = []

    # -- 上下文派生 ----------------------------------------------------------
    @property
    def active_doc(self) -> str:
        """当前活跃文档（最近一次命中的源文件）。"""
        for t in reversed(self.turns):
            if t.doc:
                return t.doc
        return ""

    @property
    def active_company(self) -> str:
        """当前话题公司（最近一轮出现的公司别名）。"""
        for t in reversed(self.turns):
            for e in t.entities:
                for _src, aliases in config.DOC_COMPANY.items():
                    if e in aliases:
                        return e
        return ""

    def topic_entities(self, window: int = 3) -> List[str]:
        """最近 window 轮的话题实体（用于检索加权）。"""
        ents: List[str] = []
        for t in self.turns[-window:]:
            for e in t.entities:
                if e not in ents:
                    ents.append(e)
        return ents

    def history_text(self, window: int = 4) -> str:
        """最近若干轮的问答文本，供界面展示或 LLM 改写参考。"""
        lines: List[str] = []
        for t in self.turns[-window:]:
            lines.append(f"Q: {t.question}")
            if t.answer:
                lines.append(f"A: {t.answer}")
        return "\n".join(lines)

    def as_messages(self, window: int = 6) -> List[dict]:
        """导出为对话消息列表（供前端渲染）。"""
        msgs: List[dict] = []
        for t in self.turns[-window:]:
            msgs.append({"role": "user", "content": t.question})
            if t.answer:
                msgs.append({"role": "assistant", "content": t.answer,
                             "rewritten": t.rewritten, "doc": t.doc,
                             "evidence": t.evidence})
        return msgs