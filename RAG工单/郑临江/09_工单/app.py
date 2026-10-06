# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务
主程序：对比基线 GraphRAG 与优化后 GraphRAG，
使用 RAGAS 上下文精度/召回指标给出优化前后变化。
用法：python app.py
"""
import glob
import os

import config
from graph_builder import build_graph_baseline, build_graph_optimized
from graph_rag import GraphRAG
from evaluation import evaluate
from llm import LLM


def load_documents():
    docs = []
    if os.path.isdir(config.CCF_TXT_DIR):
        for f in sorted(glob.glob(os.path.join(config.CCF_TXT_DIR, "*.txt"))):
            with open(f, encoding="utf-8", errors="ignore") as fp:
                docs.append(fp.read())
    else:
        import pymupdf
        for f in sorted(glob.glob(os.path.join(config.CCF_PDF_DIR, "*.pdf"))):
            d = pymupdf.open(f)
            docs.append("\n".join(p.get_text() for p in d))
            d.close()
    return docs


def main():
    docs = load_documents()
    llm = LLM()

    print("[GraphRAG优化] 构建基线图谱...")
    base_graph = build_graph_baseline(docs)
    print("[GraphRAG优化] 构建优化图谱（prompt 引导 LLM 抽取）...")
    opt_graph = build_graph_optimized(docs, llm)

    base_rag = GraphRAG(base_graph, docs)
    opt_rag = GraphRAG(opt_graph, docs)

    print("\n%-45s %-18s %-18s" % ("问题", "基线(精度/召回)", "优化(精度/召回)"))
    for pair in config.QA_PAIRS:
        b_ctx = base_rag.retrieve(pair["q"])
        o_ctx = opt_rag.retrieve(pair["q"])
        b = evaluate(b_ctx, pair["a"])
        o = evaluate(o_ctx, pair["a"])
        print("%-45s %-18s %-18s" % (
            pair["q"][:30],
            f"{b['context_precision']}/{b['context_recall']}",
            f"{o['context_precision']}/{o['context_recall']}",
        ))

    print("\n优化说明：")
    print("1. 实体/关系抽取：由粗粒度共现改为 prompt 引导 LLM 精确抽取（类型限定）")
    print("2. 检索层面：图谱邻居扩展 + 文本检索联合召回")
    print("3. 目标：context precision >= 0.8，context recall >= 0.9")


if __name__ == "__main__":
    main()
