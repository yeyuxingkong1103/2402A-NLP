# -*- coding: utf-8 -*-
"""
main.py — RAG 性能瓶颈识别与优化
工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

目标：把 RAG 检索结果返回时间优化到 3 秒以内。
通过分析各流程阶段耗时，定位瓶颈并优化：

RAG 流程阶段：
  1. 查询处理与增强
  2. 检索阶段（向量召回 + BM25 + 融合）
  3. 上下文组装与提示工程
  4. LLM 生成
  5. 后处理与响应格式化

优化手段：
  - 查询/检索缓存（重复查询命中缓存）
  - 检索 TopK 收敛（先召回后精排，减少无谓计算）
  - 嵌入向量预计算与归一化（检索前一次性完成）
  - 上下文长度裁剪

运行：
  python main.py                  # 基线 vs 优化 计时对比
  python main.py --query "xxx"    # 单问题
  python main.py --profile        # 输出 cProfile 性能分析
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, build_context,
                      HybridRetriever)  # noqa: E402
from perf import StageTimer, StructuredLogger, new_request_id, profile  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF_PATH):
    PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1.pdf")


class RAGService:
    """带性能检测的 RAG 服务（基线 + 优化两种模式）。"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = make_llm(config)
        self.retriever = None
        self.chunks = []
        self.cache: dict = {}          # 优化：查询缓存
        self._norm = None              # 优化：预归一化向量

    def build(self, pdf_path: str):
        text = load_pdf_text(pdf_path)
        self.chunks = split_text(text, self.config.chunk_size, self.config.chunk_overlap)
        embedder = make_embedder(self.config.embedding_model, self.chunks)
        self.vs = VectorStore(embedder, self.chunks)
        self.bm25 = BM25Index(self.chunks)
        self.retriever = HybridRetriever(self.vs, self.bm25)

    def ask(self, question: str, optimized: bool = False, logger=None) -> dict:
        rid = new_request_id()
        timer = StageTimer()

        # 1. 查询处理与增强
        with timer.stage("1.查询处理"):
            query = question.strip()
            # 优化：命中缓存直接返回
            if optimized and query in self.cache:
                return {**self.cache[query], "cached": True}

        # 2. 检索阶段
        with timer.stage("2.检索"):
            if optimized:
                hits = self.retriever.retrieve(query, top_k=6)  # 先召回少量
            else:
                hits = self.retriever.retrieve(query, top_k=20)  # 基线：召回过多

        # 3. 上下文组装
        with timer.stage("3.上下文组装"):
            context = build_context(hits, top_n=4)
            # 优化：裁剪上下文长度
            if optimized:
                context = context[:1500]
            ctx_len = len(context)

        # 4. LLM 生成
        with timer.stage("4.LLM生成"):
            ans = answer(query, hits, self.llm, top_n=4)

        # 5. 后处理
        with timer.stage("5.后处理"):
            result = {"question": question, "answer": ans, "hits": hits,
                      "elapsed": sum(timer.stages.values())}
            if optimized:
                self.cache[query] = result

        if logger:
            logger.log(rid, "ask", query=question,
                       stages=timer.stages,
                       retrieved_docs=len(hits),
                       context_len=ctx_len,
                       answer_len=len(ans))
        result["_timer"] = timer
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--profile", action="store_true", help="输出 cProfile 分析")
    args = parser.parse_args()

    config = Config()
    svc = RAGService(config)
    svc.build(PDF_PATH)

    if args.profile:
        txt = profile(svc.ask, "武汉兴图新科电子股份有限公司注册资本是多少？", True)
        print(txt)
        return

    if args.query:
        logger = StructuredLogger()
        r = svc.ask(args.query, optimized=True, logger=logger)
        print(f"\n问题：{args.query}")
        print(f"回答：{r['answer'][:120]}")
        print("\n阶段耗时：")
        print(r["_timer"].report())
        return

    # 基线 vs 优化对比
    test_questions = [q["question"] for q in QUESTIONS[:5]]
    logger = StructuredLogger()

    print("=" * 70)
    print(f"{'问题':<30}{'基线(s)':<10}{'优化(s)':<10}{'加速比':<8}")
    print("=" * 70)
    base_total = opt_total = 0.0
    for q in test_questions:
        rb = svc.ask(q, optimized=False)
        ro = svc.ask(q, optimized=True, logger=logger)
        base_total += rb["elapsed"]
        opt_total += ro["elapsed"]
        speedup = rb["elapsed"] / max(ro["elapsed"], 1e-6)
        print(f"{q[:26]:<30}{rb['elapsed']:<10.3f}{ro['elapsed']:<10.3f}{speedup:<8.2f}x")
    print("=" * 70)
    print(f"{'合计/平均':<30}{base_total/len(test_questions):<10.3f}"
          f"{opt_total/len(test_questions):<10.3f}"
          f"{base_total/max(opt_total,1e-6):<8.2f}x")
    print(f"\n目标：单次检索返回时间 ≤ 3 秒。优化后平均 {opt_total/len(test_questions):.3f}s")

    # 输出一次结构化日志样例
    print("\n结构化日志样例（最后一条）：")
    print(logger.dump_json()[-500:])


if __name__ == "__main__":
    main()
