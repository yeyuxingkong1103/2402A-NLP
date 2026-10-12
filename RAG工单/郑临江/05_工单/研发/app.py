# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
主程序：在招股说明书检索问答基础上实现多轮对话（含指代消解）。
用法：python app.py
"""
import config
from pdf_parser import load_pdf_text, chunk_text
from retriever import BM25Retriever
from llm import LLM, build_rag_prompt
from query_understanding import understand


class MultiTurnQA:
    def __init__(self):
        self.llm = LLM()
        self.retriever = BM25Retriever()
        self.history = []  # [(原始问题, 改写后问题, 答案)]

    def build(self, pdf_paths):
        all_docs = []
        for pdf in pdf_paths:
            print(f"[RAG] 解析 PDF：{pdf}")
            text = load_pdf_text(pdf)
            all_docs.extend(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
        self.retriever.build_index(all_docs)
        print(f"[RAG] 索引构建完成，共 {len(all_docs)} 块。")

    def ask(self, question):
        resolved, intent, entity = understand(question, self.history)
        hits = self.retriever.search(resolved, top_k=config.TOP_K)
        answer = self.llm.generate(build_rag_prompt(resolved, hits))
        self.history.append((question, resolved, answer))
        return {"raw": question, "resolved": resolved, "intent": intent,
                "entity": entity, "answer": answer, "hits": hits}


def run_demo(qa):
    print("\n================ 多轮对话演示 ================")
    for q in config.DIALOGUE:
        res = qa.ask(q)
        print(f"\n用户：{res['raw']}")
        print(f"理解：{res['resolved']}  （意图={res['intent']}，实体={res['entity']}）")
        print(f"回答：{res['answer'][:150].replace(chr(10), ' ')}")
    print("\n================ 演示结束 ================")


if __name__ == "__main__":
    qa = MultiTurnQA()
    qa.build([config.PDF1, config.PDF2])
    run_demo(qa)
