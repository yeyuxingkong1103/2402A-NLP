# -*- coding: utf-8 -*-
"""多轮问答引擎：串联 Query 改写 → 检索 → 重排 → 答案合成。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

对外主接口 `MultiTurnQA.ask(question)`：
    1. 读取会话上下文，调用 QueryRewriter 把口语问句改写为独立问句；
    2. 用改写后的问句做混合检索（向量 + BM25 + RRF），并注入话题实体；
    3. 多信号重排后抽取答案；
    4. 记录本轮 Turn（实体 / 意图 / 证据），供下一轮改写使用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from src import config
from src.answer_builder import Answer, build_answer
from src.conversation import Conversation, Turn
from src.knowledge_base import KnowledgeBase, load_kb
from src.query_rewriting import QueryRewriter, make_intent
from src.query_understanding import QueryUnderstanding
from src.reranker import rerank
from src.retriever import Retriever


@dataclass
class QAResult:
    """单轮问答结果（含调试信息）。"""
    question: str
    rewritten: str
    answer: str
    answer_type: str
    doc: str
    latency: float
    rewrite_reasons: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    evidence: List[dict] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)


class MultiTurnQA:
    """多轮检索问答引擎。"""

    def __init__(self, kb: KnowledgeBase | None = None, use_llm_rewrite: bool | None = None):
        self.kb = kb or load_kb()
        self.retriever = Retriever(self.kb)
        self.qu = QueryUnderstanding()
        self.rewriter = QueryRewriter()
        self.use_llm_rewrite = (config.USE_LLM_REWRITE if use_llm_rewrite is None
                                else use_llm_rewrite)
        self._llm = None

    def warmup(self) -> None:
        """预热：提前加载 BM25 / jieba 分词与向量编码器，避免首问冷启动超过 3s。"""
        try:
            self.retriever.search("预热查询", top_k=1)
        except Exception:
            pass

    # -- 主流程 --------------------------------------------------------------
    def ask(self, question: str, conv: Conversation) -> QAResult:
        t0 = time.time()

        # 1) 多轮 Query 改写
        rw = self.rewriter.rewrite(question, conv)
        if self.use_llm_rewrite and rw.need_rewrite:
            rw = self._llm_refine(rw, conv)

        # 2) 以改写后的独立问句做 Query 理解
        analysis = self.qu.analyze(rw.rewritten or question)
        # 话题实体注入检索
        topics = conv.topic_entities(window=2)
        extra = list(analysis.expand_queries)
        if topics:
            extra.append(" ".join(topics) + " " + " ".join(analysis.keywords[:6]))

        # 3) 混合检索 + 重排
        cands = self.retriever.search(analysis.core or rw.rewritten, top_k=config.ANSWER_POOL,
                                      extra_queries=extra)
        ranked = rerank(self.kb, cands, analysis, topic_entities=topics,
                        top_k=config.ANSWER_TOP_K)

        # 4) 答案合成（相关性兜底：最高分过低 → 视为与文档无关）
        if not ranked or ranked[0].score < config.MIN_ANSWER_SCORE:
            ans = Answer(
                text="未在招股说明书中检索到与该问题相关的内容，请尝试换一种与"
                     "公司/财务/业务相关的问法。",
                answer_type="out_of_domain", evidence=[], doc="", confidence=0.0)
        else:
            ans: Answer = build_answer(self.kb, ranked, analysis)

        latency = time.time() - t0
        result = QAResult(
            question=question,
            rewritten=rw.rewritten,
            answer=ans.text,
            answer_type=analysis.answer_type,
            doc=ans.doc,
            latency=round(latency, 3),
            rewrite_reasons=rw.reasons,
            entities=analysis.entities,
            evidence=ans.evidence,
            candidates=[{"chunk_id": r.chunk_id, "score": round(r.score, 4),
                         "source": self.kb.chunks[r.chunk_id].source,
                         "page": self.kb.chunks[r.chunk_id].page,
                         "kind": self.kb.chunks[r.chunk_id].kind}
                        for r in ranked],
        )

        # 5) 写入会话记忆
        conv.add(Turn(
            index=len(conv.turns) + 1,
            question=question,
            rewritten=rw.rewritten,
            answer=ans.text,
            doc=ans.doc or rw.doc_hint,
            answer_type=analysis.answer_type,
            entities=analysis.entities,
            keywords=analysis.keywords,
            intent=make_intent(rw.rewritten or question),
            evidence=ans.evidence,
        ))
        return result

    # -- 可选：本地 LLM 精修改写 ---------------------------------------------
    def _llm_refine(self, rw, conv: Conversation):
        """用本地大模型对规则改写结果做二次确认/修正（默认关闭）。"""
        try:
            import requests
            prompt = (
                "下面是一段多轮问答历史，以及用户的最新问题。请把最新问题改写为一个"
                "语义完整、可独立检索的问题，只输出改写结果，不要解释。\n\n"
                f"历史：\n{conv.history_text()}\n\n最新问题：{rw.original}\n改写："
            )
            r = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1}},
                timeout=config.LLM_TIMEOUT,
            )
            text = (r.json().get("response") or "").strip().splitlines()
            text = text[-1].strip() if text else ""
            if 4 <= len(text) <= 80:
                rw.rewritten = text
                rw.reasons.append("LLM 改写精修")
        except Exception:
            pass
        return rw