# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：主系统（多文档 + 表格优化版）
功能：多PDF加载 + 混合检索 + Reranker 精排
"""

import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from pdf_parser import PDFParser
from image_chunks import get_image_chunks
from vector_retriever import VectorRetriever
from query_understanding import QueryUnderstanding
from llm_generator import LLMGenerator
from rag_evaluator import RAGEvaluator


class RAGQASystem:
    """RAG 问答系统（多文档 + 表格优化版）"""

    def __init__(self, pdf_paths, llm_api_key: str = None):
        """
        :param pdf_paths: 可以是单个路径（str）或列表（List[str]）
        """
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        print("=" * 60)
        print(f"正在初始化 RAG 系统（多文档版），共 {len(pdf_paths)} 个文档")
        print("=" * 60)

        all_text_chunks = []
        all_table_chunks = []

        # 逐文档解析
        for path in pdf_paths:
            parser = PDFParser(path)
            pages = parser.extract_text()
            text_chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)
            table_chunks = parser.extract_table_chunks()
            print(f"✅ {parser.doc_name}：{len(pages)} 页 / "
                  f"{len(text_chunks)} 文本块 / {len(table_chunks)} 表格块")
            all_text_chunks.extend(text_chunks)
            all_table_chunks.extend(table_chunks)

        image_chunks = get_image_chunks()
        self.chunks = all_text_chunks + all_table_chunks + image_chunks
        print(f"✅ 总知识库：{len(self.chunks)} 块（文本 {len(all_text_chunks)} + 表格 {len(all_table_chunks)} + 图像 {len(image_chunks)}）")

        # 向量 + BM25 索引
        self.retriever = VectorRetriever(model_name="BAAI/bge-m3")
        self.retriever.build_index(self.chunks)

        # Reranker
        print("正在加载 Reranker：BAAI/bge-reranker-base")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("✅ Reranker 加载完成")

        # 其他模块
        self.query_understanding = QueryUnderstanding()
        self.llm_generator = LLMGenerator(api_key=llm_api_key)
        self.evaluator = RAGEvaluator()

        print("=" * 60)
        print("✅ 系统初始化完成")
        print("=" * 60)

    def retrieve_and_rerank(self, question: str,
                             top_k_retrieve: int = 20,
                             top_k_final: int = 3) -> List[Dict[str, Any]]:
        """混合检索 Top-20 → Reranker 精排 → Top-3"""
        candidates = self.retriever.retrieve(question, top_k=top_k_retrieve)
        if not candidates:
            return []
        pairs = [[question, c["content"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k_final]

    def answer(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """回答问题"""
        t0 = time.time()
        qinfo = self.query_understanding.process(question)
        result = {"question": question, "query_info": qinfo, "use_rag": use_rag}

        if use_rag:
            contexts = self.retrieve_and_rerank(question, top_k_retrieve=20, top_k_final=3)
            result["retrieved_contexts"] = contexts
            result["answer"] = self.llm_generator.generate_with_rag(question, contexts)
        else:
            result["answer"] = self.llm_generator.generate_without_rag(question)

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
    system = RAGQASystem([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ])
    r = system.answer("武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？", use_rag=True)
    print("=" * 60)
    print("问题：", r["question"])
    print("答案：")
    print(r["answer"])
    print(f"响应时间：{r['response_time']}s")
    print("=" * 60)
