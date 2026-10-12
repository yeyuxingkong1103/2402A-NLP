# -*- coding: utf-8 -*-
"""
main.py — 基于 PDF 文档的问答系统（RAG）
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统

功能：
  1. 解析《招股说明书1.pdf》并构建向量索引
  2. 对 10 个测试问题进行 RAG 检索 + LLM 生成回答
  3. 对比「RAG 回答」与「仅用 LLM 回答」的效果
  4. 使用 RAGAS 风格指标输出评估结果

运行：
  python main.py                # 构建索引并对全部 10 个问题问答
  python main.py --query "xxx"  # 对单个问题问答
"""
from __future__ import annotations

import argparse
import os
import time

import rag_core
from rag_core import (Config, HybridRetriever, BM25Index, VectorStore,
                      load_pdf_text, split_text, make_embedder, make_llm,
                      answer, evaluate_rag, build_context, try_ragas_eval)
from questions import QUESTIONS
from query_understanding import understand

# 附件路径（工单目录的上级「附件」目录）
ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF_PATH):
    PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1.pdf")


class RAGPipeline:
    """RAG 问答流水线。"""

    def __init__(self, config: Config):
        self.config = config
        self.chunks: list = []
        self.vector_store: VectorStore = None
        self.bm25: BM25Index = None
        self.retriever: HybridRetriever = None
        self.llm = make_llm(config)
        self.embedder_name = ""

    def build(self, pdf_path: str):
        """解析 PDF → 分块 → 建索引。"""
        print(f"[1/3] 解析 PDF：{pdf_path}")
        text = load_pdf_text(pdf_path)
        print(f"      提取文本 {len(text)} 字符")

        self.chunks = split_text(text, self.config.chunk_size, self.config.chunk_overlap)
        print(f"[2/3] 分块完成：{len(self.chunks)} 个 chunk（size={self.config.chunk_size}, overlap={self.config.chunk_overlap}）")

        embedder = make_embedder(self.config.embedding_model, self.chunks)
        self.embedder_name = embedder.name
        self.vector_store = VectorStore(embedder, self.chunks)
        self.bm25 = BM25Index(self.chunks)
        self.retriever = HybridRetriever(self.vector_store, self.bm25)
        print(f"[3/3] 索引构建完成（嵌入器：{self.embedder_name}）")

    def ask(self, question: str, top_k: int = 5) -> dict:
        """单次 RAG 问答：Query 理解 → 检索 → 生成。"""
        t0 = time.time()
        # 1. Query 理解
        qu = understand(question, self.llm)
        # 2. 检索（用规范化后的主问题）
        hits = self.retriever.retrieve(qu["query"], top_k=top_k)
        # 3. 生成
        ans = answer(qu["query"], hits, self.llm, top_n=4)
        elapsed = time.time() - t0
        return {
            "question": question,
            "intent": qu["intent"],
            "hits": hits,
            "answer": ans,
            "elapsed": elapsed,
        }


def run_batch(pipeline: RAGPipeline):
    """对 10 个测试问题批量问答，输出对比与评估。"""
    pure_llm = rag_core.make_llm(pipeline.config)
    print("\n" + "=" * 70)
    print("RAG 检索问答 vs 纯 LLM 回答对比")
    print("=" * 70)

    rows = []
    for q in QUESTIONS:
        res = pipeline.ask(q["question"])
        # 纯 LLM 回答（不带检索上下文）
        pure_ans = pure_llm.generate(f"请直接回答以下问题：{q['question']}")

        print(f"\n【问题 {q['id']}】{q['question']}")
        print(f"  [意图] {res['intent']}   [耗时] {res['elapsed']:.2f}s")
        print(f"  [RAG 回答] {res['answer'][:120]}")
        print(f"  [纯LLM回答] {pure_ans[:120]}")

        ctx = build_context(res["hits"], top_n=4)
        metrics = evaluate_rag(q["question"], "", ctx, res["answer"])
        print(f"  [评估] 上下文精度={metrics['context_precision']:.3f} "
              f"召回={metrics['context_recall']:.3f} 忠实度={metrics['faithfulness']:.3f}")

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": res["answer"],
            "contexts": [h["text"] for h in res["hits"][:4]],
            "reference": "",
        })

    # 若安装了 ragas，输出真实 RAGAS 指标
    ragas_result = try_ragas_eval(rows)
    if ragas_result:
        print("\n[RAGAS 评估结果]")
        for k, v in ragas_result.items():
            print(f"  {k}: {v}")
    else:
        print("\n（未安装 ragas，已使用内置离线指标完成评估。安装：pip install ragas）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="单个问题")
    args = parser.parse_args()

    config = Config()
    pipeline = RAGPipeline(config)
    pipeline.build(PDF_PATH)

    if args.query:
        res = pipeline.ask(args.query)
        print(f"\n问题：{res['question']}")
        print(f"意图：{res['intent']}  耗时：{res['elapsed']:.2f}s")
        print(f"回答：{res['answer']}")
        print("\n检索到的相关片段：")
        for i, h in enumerate(res["hits"][:4], 1):
            print(f"  [{i}] ({h['score']:.3f}) {h['text'][:80]}...")
    else:
        run_batch(pipeline)


if __name__ == "__main__":
    main()
