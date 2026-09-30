# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：主系统（优化版）
"""

import time
from typing import List, Dict, Any

from pdf_parser import PDFParser
from vector_retriever import VectorRetriever
from query_understanding import QueryUnderstanding
from llm_generator import LLMGenerator
from rag_evaluator import RAGEvaluator


class RAGQASystem:
    """RAG 问答系统（优化版）"""

    def __init__(self, pdf_path: str, llm_api_key: str = None):
        print("=" * 60)
        print("正在初始化 RAG 系统...")
        print("=" * 60)

        self.parser = PDFParser(pdf_path)
        pages = self.parser.extract_text()
        tables = self.parser.extract_tables()
        print(f"✅ PDF 解析：{len(pages)} 页，{len(tables)} 个表格")

        self.chunks = self.parser.chunk_text(pages, chunk_size=300, overlap=80)
        print(f"✅ 文本切分：{len(self.chunks)} 块")

        table_chunks = self._tables_to_chunks(tables)
        self.chunks.extend(table_chunks)
        print(f"✅ 表格块加入：{len(table_chunks)} 块，总计 {len(self.chunks)} 块")

        self.retriever = VectorRetriever(model_name="BAAI/bge-base-zh-v1.5")
        self.retriever.build_index(self.chunks)

        self.query_understanding = QueryUnderstanding()
        self.llm_generator = LLMGenerator(api_key=llm_api_key)
        self.evaluator = RAGEvaluator()

        print("=" * 60)
        print("✅ 系统初始化完成")
        print("=" * 60)

    def _tables_to_chunks(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for t in tables:
            rows = t.get("data", [])
            if not rows:
                continue
            lines = []
            for row in rows:
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            text = "[表格] " + "\n".join(lines)
            if len(text) > 20:
                chunks.append({
                    "chunk_id": f"table_p{t['page']}_{t.get('table_index', 0)}",
                    "content": text,
                    "page": t["page"],
                    "source": t.get("source", f"table_p{t['page']}")
                })
        return chunks

    def answer(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        qinfo = self.query_understanding.process(question)
        result = {"question": question, "query_info": qinfo, "use_rag": use_rag}

        if use_rag:
            contexts = self.retriever.retrieve(question, top_k=10)
            result["retrieved_contexts"] = contexts
            answer = self.llm_generator.generate_with_rag(question, contexts)
            result["answer"] = answer
            result["retrieval_eval"] = self.evaluator.evaluate_retrieval(question, contexts)
            result["answer_eval"] = self.evaluator.evaluate_answer_relevance(question, answer, contexts)
        else:
            answer = self.llm_generator.generate_without_rag(question)
            result["answer"] = answer
            result["answer_eval"] = self.evaluator.evaluate_answer_relevance(question, answer, [])

        elapsed = time.time() - t0
        result["response_time"] = round(elapsed, 3)
        result["within_3s"] = elapsed <= 3.0
        return result

    def batch_evaluate(self, questions: List[Dict]) -> List[Dict]:
        results = []
        for q in questions:
            question = q["question"]
            rag_r = self.answer(question, use_rag=True)
            llm_r = self.answer(question, use_rag=False)
            cmp = self.evaluator.compare_with_llm_only(
                question, rag_r["answer"], llm_r["answer"]
            )
            results.append({
                "id": q.get("id"),
                "question": question,
                "rag_answer": rag_r["answer"],
                "llm_answer": llm_r["answer"],
                "rag_response_time": rag_r["response_time"],
                "llm_response_time": llm_r["response_time"],
                "comparison": cmp,
                "retrieval_eval": rag_r.get("retrieval_eval", {})
            })
        return results


if __name__ == "__main__":
    system = RAGQASystem("./data/招股说明书1.pdf")
    r = system.answer("武汉兴图新科电子股份有限公司注册资本是多少？", use_rag=True)
    print("=" * 60)
    print("问题：", r["question"])
    print("答案：")
    print(r["answer"])
    print(f"响应时间：{r['response_time']}s")
