# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
多轮对话链：Query改写（指代消解/省略补全） → 检索 → 生成 → 维护对话历史。
对比开关 use_rewrite：关闭时直接用原问句检索，用于验证Query理解优化的必要性。
"""
from query_rewriter import QueryRewriter
from rag_chain import (
    retrieve_contexts, call_llm, build_rag_prompt, RAG_SYSTEM_PROMPT,
)


class DialogSession:
    """一次多轮对话会话"""

    def __init__(self, retriever, use_rewrite=True, top_k=5):
        self.retriever = retriever
        self.rewriter = QueryRewriter() if use_rewrite else None
        self.use_rewrite = use_rewrite
        self.top_k = top_k
        self.history = []      # [{"role","content"}]
        self.turn_log = []     # 逐轮记录，供报告使用

    def ask(self, question):
        """处理一轮对话，返回 dict(question, rewritten, answer, contexts, time)"""
        rewritten = self.rewriter.rewrite(question, self.history) if self.rewriter else question

        ctxs = retrieve_contexts(rewritten, self.retriever, top_k=self.top_k,
                                 optimize=True, use_rerank=True)
        answer = call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(rewritten, ctxs))

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        rec = {
            "turn": len(self.turn_log) + 1,
            "question": question,
            "rewritten": rewritten,
            "answer": answer,
            "pages": [c["page"] for c, _ in ctxs],
            "docs": sorted({c["doc"] for c, _ in ctxs}),
        }
        self.turn_log.append(rec)
        return rec


def run_dialog(questions, retriever, use_rewrite=True, verbose=True):
    """按给定问题序列跑一轮多轮对话，返回逐轮记录"""
    sess = DialogSession(retriever, use_rewrite=use_rewrite)
    for q in questions:
        rec = sess.ask(q)
        if verbose:
            print(f"\n[第{rec['turn']}轮] 用户: {rec['question']}")
            if rec["rewritten"] != rec["question"]:
                print(f"  → 改写: {rec['rewritten']}")
            print(f"  → 答案: {rec['answer'][:200]}")
    return sess.turn_log
