# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
测试主程序：对 ccf_competition 金融年报执行 RAG 检索测试，
输出 10 个问题及对应的检索结果、评估结果，并分析存在的问题。
"""
import os
import glob
import json

import config
from retriever import BM25Retriever
from evaluation import evaluate


def load_documents():
    """加载 ccf 年报文本（优先 txt，其次解析 pdf）。"""
    docs = []
    if os.path.isdir(config.CCF_TXT_DIR):
        files = sorted(glob.glob(os.path.join(config.CCF_TXT_DIR, "*.txt")))
        for f in files:
            with open(f, encoding="utf-8", errors="ignore") as fp:
                docs.append(fp.read())
    elif os.path.isdir(config.CCF_PDF_DIR):
        import pymupdf
        files = sorted(glob.glob(os.path.join(config.CCF_PDF_DIR, "*.pdf")))
        for f in files:
            doc = pymupdf.open(f)
            docs.append("\n".join(p.get_text() for p in doc))
            doc.close()
    return docs


def main():
    docs = load_documents()
    print(f"[TEST] 加载年报文档 {len(docs)} 篇")
    retriever = BM25Retriever()
    retriever.build_index(docs)

    results = []
    for q in config.QUESTIONS:
        hits = retriever.search(q["question"], top_k=config.TOP_K)
        context = " ".join(c for c, _ in hits)
        # 检索式“答案”（降级模式：直接用上下文作为答案进行评测）
        answer = context[:200]
        metrics = evaluate(q["question"], answer, hits, config.REFERENCE_ANSWERS.get(q["id"], ""))
        results.append({
            "id": q["id"], "question": q["question"],
            "answer": answer, "metrics": metrics,
            "hits": [{"text": c[:120], "score": round(s, 4)} for c, s in hits],
        })
        print(f"问题{q['id']} 完成，指标={metrics}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已写入 results.json")


if __name__ == "__main__":
    main()
