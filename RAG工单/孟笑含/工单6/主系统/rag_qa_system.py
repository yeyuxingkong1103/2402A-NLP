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
from hybrid_retriever import HybridRetriever
from retrieval_config import RetrievalConfig
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

        # 混合检索（可配置：向量 / 全文 / 混合）
        self.retrieval_config = RetrievalConfig(
            mode="hybrid",
            use_rerank=True,
            rerank_method="cross_encoder",
        )
        self.retriever = HybridRetriever(self.retrieval_config)
        self.retriever.build_index(self.chunks)

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
        """混合检索（含 Reranker 精排）"""
        return self.retriever.retrieve(question, top_k=top_k_final)

    def set_retrieval_mode(self, mode: str, use_rerank: bool = None,
                            rerank_method: str = None):
        """动态切换检索模式（工单6 新增）"""
        if mode in ["vector", "fulltext", "hybrid"]:
            self.retrieval_config.mode = mode
        if use_rerank is not None:
            self.retrieval_config.use_rerank = use_rerank
        if rerank_method is not None:
            self.retrieval_config.rerank_method = rerank_method
            self.retriever.reranker = None  # 强制重载
        print(f"✅ 检索模式切换：mode={self.retrieval_config.mode}, "
              f"use_rerank={self.retrieval_config.use_rerank}, "
              f"rerank_method={self.retrieval_config.rerank_method}")

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


# ==================== 工单5 新增：多轮对话接口 ====================

def _extract_company(text: str):
    """从问题里提取公司名"""
    import re
    companies = [
        "武汉力源信息技术股份有限公司", "武汉力源",
        "武汉兴图新科电子股份有限公司", "武汉兴图新科", "兴图新科",
    ]
    for c in companies:
        if c in text:
            return c
    return None


def answer_with_history(self, question: str, conversation) -> dict:
    """多轮对话接口（工单5 新增）

    :param question: 用户本轮问题
    :param conversation: Conversation 实例
    :return: 标准 answer() 字典 + rewritten_query 字段
    """
    from conversation import Conversation
    if conversation is None:
        conversation = Conversation()

    # 1. 指代消解改写
    rewritten = self.query_understanding.rewrite_with_context(question, conversation)

    # 2. 正常 RAG 问答
    result = self.answer(rewritten, use_rag=True)
    result["original_question"] = question
    result["rewritten_query"] = rewritten

    # 3. 更新会话状态
    company = _extract_company(rewritten)
    if company:
        # 统一存全称
        if "力源" in company:
            conversation.set_company("武汉力源信息技术股份有限公司")
        else:
            conversation.set_company("武汉兴图新科电子股份有限公司")

    # 从问题里提取"问题类型"（用于"那XXX呢"追问）
    qtype_keywords = {
        "法定代表人": "法定代表人",
        "注册资本": "注册资本",
        "技术标准": "参与制定的技术标准",
        "发行股数": "发行股数",
        "募集资金": "募集资金投资项目",
        "关联方": "关联方",
        "上游": "行业上游",
        "下游": "行业下游",
        "一等奖": "荣获国家科技进步一等奖的工程",
        "军用领域收入": "军用领域收入",
    }
    for kw, qtype in qtype_keywords.items():
        if kw in rewritten:
            conversation.set_question_type(qtype)
            break

    # 4. 记录对话
    conversation.add(question, result["answer"], result.get("query_info"))

    return result


# 绑定到 RAGQASystem 类
RAGQASystem.answer_with_history = answer_with_history
