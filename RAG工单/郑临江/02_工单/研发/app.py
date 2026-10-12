# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
主程序：使用“优化方案”（句子感知分块 + BM25 检索）回答问题。
用法：
  python app.py            # 交互式提问
  python app.py --test     # 批量运行 10 个问题并展示优化后答案
"""
import sys
import time

import config
from pdf_parser import load_pdf_text, chunk_sentence
from retriever import BM25Retriever
from llm import LLM, build_rag_prompt


class OptimizedQA:
    def __init__(self):
        self.llm = LLM()
        self.retriever = BM25Retriever()

    def build(self, pdf_path):
        print(f"[优化RAG] 解析 PDF：{pdf_path}")
        text = load_pdf_text(pdf_path)
        chunks = chunk_sentence(text, config.OPT_CHUNK_SIZE, config.OPT_OVERLAP)
        self.retriever.build_index(chunks)
        print(f"[优化RAG] 句子感知分块完成，共 {len(chunks)} 块。")

    def ask(self, question):
        start = time.time()
        hits = self.retriever.search(question, top_k=config.TOP_K)
        answer = self.llm.generate(build_rag_prompt(question, hits))
        return {"question": question, "answer": answer,
                "elapsed": time.time() - start, "hits": hits}


def run_test(qa):
    print("\n================ 优化后问答演示 ================")
    for q in config.QUESTIONS:
        res = qa.ask(q["question"])
        print(f"\n--- 问题(id={q['id']}) ---")
        print("Q:", q["question"])
        print("A:", res["answer"][:150].replace("\n", " "))
        print(f"耗时 {res['elapsed']:.3f}s")
    print("\n================ 演示结束 ================")


def run_interactive(qa):
    print("\n进入交互式问答（输入 exit 退出）：")
    while True:
        q = input("\n请输入问题：").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        res = qa.ask(q)
        print("答案：", res["answer"])
        print(f"（耗时 {res['elapsed']:.3f}s）")


if __name__ == "__main__":
    qa = OptimizedQA()
    qa.build(config.PDF1)
    if "--test" in sys.argv:
        run_test(qa)
    else:
        run_interactive(qa)
