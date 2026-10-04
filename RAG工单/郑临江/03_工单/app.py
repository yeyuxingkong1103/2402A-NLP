# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
主程序：解析《招股说明书1.pdf》《招股说明书2.pdf》，
包含表格解析，支持表格类问题的检索回复。
用法：
  python app.py --test     # 批量运行全部问题
  python app.py --tables   # 仅展示解析出的表格（验证表格解析）
"""
import sys

import config
from pdf_parser import extract_pdf_content, chunk_text
from retriever import BM25Retriever
from llm import LLM, build_rag_prompt


class TableRagQA:
    def __init__(self):
        self.llm = LLM()
        self.retriever = BM25Retriever()
        self.tables = []

    def build(self, pdf_paths):
        all_docs = []
        for pdf in pdf_paths:
            print(f"[RAG] 解析 PDF：{pdf}")
            text_blocks, table_blocks = extract_pdf_content(pdf)
            self.tables.extend(table_blocks)
            for t in text_blocks:
                all_docs.extend(chunk_text(t, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
            # 表格整体作为一个独立检索块
            all_docs.extend(table_blocks)
        self.retriever.build_index(all_docs)
        print(f"[RAG] 索引构建完成，共 {len(all_docs)} 个检索块（含 {len(self.tables)} 个表格）。")

    def ask(self, question):
        hits = self.retriever.search(question, top_k=config.TOP_K)
        answer = self.llm.generate(build_rag_prompt(question, hits))
        return {"question": question, "answer": answer, "hits": hits}


def run_test(qa):
    print("\n================ 表格解析检索演示 ================")
    for q in config.QUESTIONS:
        res = qa.ask(q["question"])
        print(f"\n--- 问题(id={q['id']}) ---")
        print("Q:", q["question"])
        print("A:", res["answer"][:200].replace("\n", " "))
    print("\n================ 演示结束 ================")


def show_tables(qa):
    print(f"\n共解析出 {len(qa.tables)} 个表格：")
    for i, t in enumerate(qa.tables):
        print(f"\n===== 表格 {i + 1} =====")
        print(t[:600])


if __name__ == "__main__":
    qa = TableRagQA()
    qa.build([config.PDF1, config.PDF2])
    if "--tables" in sys.argv:
        show_tables(qa)
    else:
        run_test(qa)
