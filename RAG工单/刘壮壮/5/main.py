# -*- coding: utf-8 -*-
"""
main.py — Query 理解优化（多轮对话）
工单编号：人工智能NLP-RAG-Query 理解优化任务

在招股说明书检索问答基础上，实现多轮对话能力：
  - 指代消解：把「他 / 这个公司 / 那…呢」解析为完整实体
  - 多轮状态：记录历史对话与当前话题实体
  - 支持如下对话：
      Q: 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
      Q: 他参与的哪个工程荣获了国家科技进步一等奖？
      Q: 这个公司的法定代表人是谁？
      Q: 那武汉力源信息技术股份有限公司呢？
      Q: 武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？

运行：
  python main.py                    # 演示内置多轮对话
  python main.py --interactive      # 交互式多轮对话
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, HybridRetriever)  # noqa: E402
from coreference import resolve_coreference, track_entity  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF1 = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF1):
    PDF1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")

# 内置演示对话（工单要求）
DEMO_DIALOG = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]


class ConversationalRAG:
    """多轮 RAG 问答。"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = make_llm(config)
        self.history: list = []  # [{"query","resolved","answer","entity"}]

    def build(self, pdf_paths):
        chunks = []
        for p in pdf_paths:
            text = load_pdf_text(p)
            chunks.extend(split_text(text, self.config.chunk_size, self.config.chunk_overlap))
        embedder = make_embedder(self.config.embedding_model, chunks)
        self.vs = VectorStore(embedder, chunks)
        self.bm25 = BM25Index(chunks)
        self.retriever = HybridRetriever(self.vs, self.bm25)
        print(f"索引构建完成：{len(chunks)} 块（嵌入器 {embedder.name}）")

    def ask(self, query: str) -> dict:
        # 1. 指代消解
        resolved = resolve_coreference(query, self.history, self.llm)
        # 2. 检索
        hits = self.retriever.retrieve(resolved, top_k=5)
        # 3. 生成
        ans = answer(resolved, hits, self.llm)
        # 4. 记录实体与历史
        entity = track_entity(query, resolved, ans)
        self.history.append({"query": query, "resolved": resolved,
                             "answer": ans, "entity": entity})
        return {"query": query, "resolved": resolved, "answer": ans,
                "entity": entity, "hits": hits}


def run_demo(cr: ConversationalRAG):
    print("\n" + "=" * 70)
    print("多轮对话演示")
    print("=" * 70)
    for q in DEMO_DIALOG:
        r = cr.ask(q)
        print(f"\nQ: {r['query']}")
        if r["resolved"] != r["query"]:
            print(f"  [指代消解] → {r['resolved']}")
        print(f"A: {r['answer'][:200]}")
    print("\n" + "=" * 70)


def run_interactive(cr: ConversationalRAG):
    print("\n进入交互式多轮对话，输入 exit 退出。")
    while True:
        q = input("\n你：").strip()
        if q.lower() in ("exit", "quit", "退出"):
            break
        r = cr.ask(q)
        print(f"[消解] {r['resolved']}")
        print(f"助手：{r['answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="交互式对话")
    args = parser.parse_args()

    config = Config()
    cr = ConversationalRAG(config)
    cr.build([PDF1, PDF2])

    if args.interactive:
        run_interactive(cr)
    else:
        run_demo(cr)


if __name__ == "__main__":
    main()
