# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
演示脚本：对工单要求的检索问题列表逐一问答，结果保存到 演示结果.md
运行：python demo.py
"""
import time
from pathlib import Path

import rag

# 工单01 要求的检索问题列表（针对《招股说明书1》：武汉兴图新科电子股份有限公司）
QUESTIONS = [
    ("260", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    ("95", "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"),
    ("33", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"),
    ("34", "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"),
    ("957", "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"),
    ("793", "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"),
    ("795", "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"),
    ("543", "武汉兴图新科电子股份有限公司注册资本是多少？"),
    ("531", "武汉兴图新科电子股份有限公司法定代表人是谁？"),
]


def main():
    col = rag.get_collection()
    lines = ["# 工单01 演示结果：基于 PDF 文档的问答系统", "",
             f"模型：BGE-M3 向量检索（ChromaDB）+ Ollama 本地 {rag.GEN_MODEL} 生成，Top-{rag.TOP_K}",
             ""]
    for qid, q in QUESTIONS:
        t0 = time.time()
        ans, hits = rag.answer(col, q)
        dt = time.time() - t0
        pages = ", ".join(str(h["page"]) for h in hits[:3])
        print(f"[{qid}] {dt:.1f}s  {q}\n  -> {ans[:80]}\n")
        lines += [f"## 问题 id={qid}", f"**Q**：{q}", "", f"**A**：{ans}", "",
                  f"耗时 {dt:.1f}s；引用页码：{pages}；最高相似度 {hits[0]['score']}" if hits else "",
                  ""]
    Path("演示结果.md").write_text("\n".join(lines), encoding="utf-8")
    print("已保存 -> 演示结果.md")


if __name__ == "__main__":
    main()
