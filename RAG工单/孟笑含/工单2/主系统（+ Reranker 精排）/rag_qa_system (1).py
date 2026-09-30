# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：主系统（优化版）
功能：整合 PDF 解析 + 混合检索 + Reranker 精排 + 答案生成
"""

import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from pdf_parser import PDFParser
from vector_retriever import VectorRetriever
from query_understanding import QueryUnderstanding
from llm_generator import LLMGenerator
from rag_evaluator import RAGEvaluator


class RAGQASystem:
    """RAG 问答系统（Reranker 精排优化版）"""

    def __init__(self, pdf_path: str, llm_api_key: str = None):
        print("=" * 60)
        print("正在初始化 RAG 系统（优化版）...")
        print("=" * 60)

        # 1. 解析 PDF
        self.parser = PDFParser(pdf_path)
        pages = self.parser.extract_text()
        tables = self.parser.extract_tables()
        print(f"✅ PDF 解析：{len(pages)} 页，{len(tables)} 个表格")

        # 2. 文本切分
        self.chunks = self.parser.chunk_text(pages, chunk_size=300, overlap=80)
        print(f"✅ 文本切分：{len(self.chunks)} 块")

        # 3. 表格加入知识库
        table_chunks = self._tables_to_chunks(tables)
        self.chunks.extend(table_chunks)
        print(f"✅ 表格块加入：{len(table_chunks)} 块，总计 {len(self.chunks)} 块")

        # 4. 混合检索（bge-m3 + BM25 + RRF）
        self.retriever = VectorRetriever(model_name="BAAI/bge-m3")
        self.retriever.build_index(self.chunks)

        # 5. Reranker 精排
        print("正在加载 Reranker 模型：BAAI/bge-reranker-base")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("✅ Reranker 加载完成")

        # 6. 其他模块
        self.query_understanding = QueryUnderstanding()
        self.llm_generator = LLMGenerator(api_key=llm_api_key)
        self.evaluator = RAGEvaluator()

        print("=" * 60)
        print("✅ 系统初始化完成")
        print("=" * 60)

        self.evaluator = RAGEvaluator()

        print("=" * 60)
        print("✅ 系统初始化完成")
        print("=" * 60)

    def _tables_to_chunks(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把表格转成文本块"""
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

    def retrieve_and_rerank(self, question: str,
                             top_k_retrieve: int = 20,
                             top_k_final: int = 3) -> List[Dict[str, Any]]:
        """混合检索 Top-20 → Reranker 精排 → Top-3"""
        # 1. 混合检索
        candidates = self.retriever.retrieve(question, top_k=top_k_retrieve)
        if not candidates:
            return []

        # 2. Reranker 打分
        pairs = [[question, c["content"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        # 3. 按 Reranker 分数排序，取 Top-K
        reranked = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k_final]

        return reranked

    def answer(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """回答问题"""
        t0 = time.time()
        qinfo = self.query_understanding.process(question)
        result = {"question": question, "query_info": qinfo, "use_rag": use_rag}

        if use_rag:
            # 混合检索 + Reranker
            contexts = self.retrieve_and_rerank(question, top_k_retrieve=20, top_k_final=3)
            result["retrieved_contexts"] = contexts
            answer = self.llm_generator.generate_with_rag(question, contexts)
            result["answer"] = answer
        else:
            answer = self.llm_generator.generate_without_rag(question)
            result["answer"] = answer

        elapsed = time.time() - t0
        result["response_time"] = round(elapsed, 3)
        result["within_3s"] = elapsed <= 3.0
        return result

    def batch_evaluate(self, questions: List[Dict]) -> List[Dict]:
        """批量对比 RAG vs 纯 LLM"""
        results = []
        for q in questions:
            question = q["question"]
            rag_r = self.answer(question, use_rag=True)
            llm_r = self.answer(question, use_rag=False)
            results.append({
                "id": q.get("id"),
                "question": question,
                "rag_answer": rag_r["answer"],
                "llm_answer": llm_r["answer"],
                "rag_response_time": rag_r["response_time"],
                "llm_response_time": llm_r["response_time"],
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
    print("=" * 60)
