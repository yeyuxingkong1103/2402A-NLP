# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
主程序：构建 RAG 问答系统。
功能：
  1) 解析《招股说明书1.pdf》并建立向量索引；
  2) 支持交互式提问与批量演示（工单要求的 10 个问题）；
  3) 对比“RAG 检索回答”与“仅 LLM 回答”。
用法：
  python app.py            # 交互式提问
  python app.py --test     # 批量运行工单问题并输出结果
"""
import sys
import time

import config
from pdf_parser import load_pdf_text, chunk_text
from retriever import TfidfRetriever
from llm import LLM, build_rag_prompt


class RagQA:
    def __init__(self):
        self.llm = LLM()
        self.retriever = TfidfRetriever()
        self.chunks = []

    def build(self, pdf_path):
        """解析 PDF 并建立索引。"""
        print(f"[RAG] 解析 PDF：{pdf_path}")
        text = load_pdf_text(pdf_path)
        self.chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.retriever.build_index(self.chunks)
        print(f"[RAG] 索引构建完成，共 {len(self.chunks)} 个文本块。")

    def ask(self, question: str, use_rag=True):
        """提问；use_rag=False 时仅用 LLM（不带检索上下文）。"""
        start = time.time()
        if use_rag:
            hits = self.retriever.search(question, top_k=config.TOP_K)
            context = "\n".join(c for c, _ in hits)
            answer = self.llm.generate(build_rag_prompt(question, hits))
        else:
            hits = []
            context = ""
            answer = self.llm.generate(f"请回答下列问题：{question}")
        elapsed = time.time() - start
        return {"question": question, "answer": answer,
                "context": context, "hits": hits, "elapsed": elapsed}


def run_test(qa):
    print("\n================ 批量演示：RAG vs 仅LLM ================")
    for q in config.QUESTIONS:
        rag = qa.ask(q["question"], use_rag=True)
        llm = qa.ask(q["question"], use_rag=False)
        print(f"\n--- 问题(id={q['id']}) ---")
        print("Q:", q["question"])
        print("RAG 回答:", rag["answer"][:120].replace("\n", " "))
        print("仅LLM回答:", llm["answer"][:120].replace("\n", " "))
        print(f"RAG 耗时: {rag['elapsed']:.3f}s")
    print("\n================ 演示结束 ================")


def run_interactive(qa):
    print("\n进入交互式问答（输入 exit 退出）：")
    while True:
        q = input("\n请输入问题：").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        res = qa.ask(q, use_rag=True)
        print("答案：", res["answer"])
        print(f"（耗时 {res['elapsed']:.3f}s）")


if __name__ == "__main__":
    qa = RagQA()
    qa.build(config.PDF1)
    if "--test" in sys.argv:
        run_test(qa)
    else:
        run_interactive(qa)
