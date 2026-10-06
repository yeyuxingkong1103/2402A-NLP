# -*- coding: utf-8 -*-
"""
问答引擎 V3 - 多文档 + 表格优先检索
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

核心改进 (对比 V2):
    1. 多文档路由: 根据公司名选择要搜索的文档
    2. 表格优先: 表格类问题优先走表格检索, 再补充文本
    3. 结构化表格→LLM: 表头+数据行+命中行高亮
    4. 非表格问题: 走 V2 混合检索 (BM25 + TF-IDF + Rerank)
"""
import os
import sys
import time
import json
import logging
from typing import Dict, List

import config_v3 as config
import multi_pdf_manager
import table_extractor
import table_retriever

# 引入 V2 模块 (复用混合检索 + LLM 调用 + 查询增强)
_V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "工单2", "研发")
_V2_DIR = os.path.abspath(_V2_DIR)
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

logger = logging.getLogger(__name__)


def _init_documents():
    """初始化: 加载所有文档的表格和文本"""
    mgr = multi_pdf_manager.get_manager()
    doc_data = {}  # doc_id -> {"tables": [...], "chunks": [...]}

    for doc in mgr.docs:
        entry = {"tables": [], "chunks": []}

        if doc.exists:
            # 从真实 PDF 提取
            tables = table_extractor.extract_tables_from_pdf(
                doc.path, doc.company, doc.company)
            entry["tables"] = tables

            chunks = table_extractor.extract_text_from_pdf(
                doc.path, doc.company, doc.company)
            entry["chunks"] = chunks
        else:
            # 降级: 加载预设表格
            preset = mgr.load_preset_tables(doc)
            if preset:
                entry["tables"] = preset
            logger.warning(f"文档缺失, 使用预设表格: {doc.company}")

        doc_data[doc.company] = entry

    return doc_data


# 全局缓存
_doc_data = None
_table_retriever = None
_text_retriever = None


def _ensure_ready():
    """确保所有索引已就绪"""
    global _doc_data, _table_retriever

    if _doc_data is None:
        _doc_data = _init_documents()

    if _table_retriever is None:
        _table_retriever = table_retriever.TableRetriever()
        # 合并所有文档的表格
        all_tables = []
        for company, entry in _doc_data.items():
            for t in entry["tables"]:
                if "company" not in t:
                    t["company"] = company
                all_tables.append(t)
        _table_retriever.load_tables(all_tables)


def answer_question(question: str) -> Dict:
    """
    V3 完整 RAG 问答

    Returns:
        {
            "question": str,
            "is_table_query": bool,
            "routed_docs": [...],
            "table_results": [...],
            "text_results": [...],
            "rag_answer": str,
            "llm_answer": str,
            "response_time": float,
            "within_time_limit": bool,
        }
    """
    start = time.time()
    _ensure_ready()

    # 1. 路由: 确定要搜哪些文档
    mgr = multi_pdf_manager.get_manager()
    routed_docs = mgr.route_query(question)
    company_filter = None
    if routed_docs and len(routed_docs) == 1:
        company_filter = routed_docs[0].company

    # 2. 判定是否表格类问题
    is_table_q = table_retriever.is_table_query(question)

    # 3. 表格检索
    table_results = []
    if is_table_q:
        table_results = _table_retriever.search(
            question, company_filter=company_filter, top_k=3)

    # 4. 文本检索 (补充)
    text_results = []
    if not is_table_q or not table_results:
        text_results = _text_search_v2(question, company_filter)

    # 5. 准备 LLM 上下文
    rag_answer = _generate_answer(question, table_results, text_results, is_table_q)

    # 6. 纯 LLM 基线
    llm_answer = _generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "is_table_query": is_table_q,
        "routed_docs": [d.company for d in routed_docs],
        "table_results": [{
            "table_name": r["table"]["table_name"],
            "company": r["table"]["company"],
            "page": r["table"].get("page"),
            "score": r["score"],
            "hit_rows": r["hit_rows"],
            "matched_cols": r["matched_cols"],
        } for r in table_results],
        "text_results": text_results[:3] if isinstance(text_results, list) else text_results,
        "rag_answer": rag_answer,
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= config.MAX_RESPONSE_TIME,
    }


def _text_search_v2(query: str, company_filter: str = None):
    """调用 V2 混合检索 (如可用)"""
    try:
        import sys
        sys.path.insert(0, _V2_DIR)
        import vector_retrieval_v2 as v2_retrieval
        retriever = v2_retrieval.get_hybrid_retriever()
        results = retriever.search(query, top_k=5)

        # 多文档时过滤
        if company_filter and results:
            results = [r for r in results
                       if company_filter in r.get("text", "") or
                       any(a in r.get("text", "") for a in
                           [d for c in multi_pdf_manager.get_manager().docs
                            if c.company == company_filter
                            for d in [c.company] + c.aliases])]
        return results
    except Exception as e:
        logger.warning(f"V2 文本检索不可用 ({e}), 返回空")
        return []


def _generate_answer(question: str, table_results: List[Dict],
                     text_results: List, is_table_query: bool) -> str:
    """生成 RAG 答案 (本地降级 + LLM)"""
    # 优先用表格检索结果
    if table_results:
        context = _table_retriever.format_for_llm(table_results)
        system = (
            "你是金融文档问答助手, 擅长从招股说明书的表格中提取关键数据。\n"
            "规则:\n"
            "1. 仅根据提供的表格数据回答, 不要编造。\n"
            "2. 回答财务数据时保留原始数字和单位。\n"
            "3. 回答结构: 先给结论, 再引用表格中的具体数据。\n"
            "4. 如果表格没有相关信息, 回答'文档中未找到相关信息'。"
        )
        user = f"请根据以下表格数据回答问题:\n\n{context}\n\n问题: {question}"
        answer = _call_llm(system, user)
        if answer:
            return answer
        # 降级: 返回表格摘要
        summaries = [r["table"]["summary"] for r in table_results[:3]]
        return "[本地模式] 表格检索结果:\n" + "\n\n".join(summaries)

    # 非表格问题: 用文本检索结果
    if text_results:
        context_parts = []
        for i, r in enumerate(text_results[:5]):
            text = r.get("text", str(r)) if isinstance(r, dict) else str(r)
            context_parts.append(f"【片段{i+1}】 {text[:300]}")
        context = "\n\n".join(context_parts)
        system = (
            "你是金融文档问答助手, 仅根据提供的招股说明书片段回答。\n"
            "如果片段没有相关信息, 回答'文档中未找到相关信息'。"
        )
        user = f"片段:\n{context}\n\n问题: {question}"
        answer = _call_llm(system, user)
        if answer:
            return answer
        return "[本地模式] 文本检索结果:\n" + context_parts[0]

    return "未检索到相关内容, 且未启用 LLM, 无法生成回答。"


def _call_llm(system: str, user: str) -> str:
    """调用 LLM (复用 V2 llm_client_v2)"""
    try:
        import sys
        sys.path.insert(0, _V2_DIR)
        import llm_client_v2
        # 用 generate_with_rag_v2 的底层函数
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        import llm_client_v2 as v2llm
        from llm_client_v2 import _call_openai_api
        return _call_openai_api(messages) or ""
    except Exception as e:
        logger.warning(f"LLM 调用不可用 ({e})")
        return ""


def _generate_without_rag(question: str) -> str:
    """纯 LLM 基线"""
    try:
        import sys
        sys.path.insert(0, _V2_DIR)
        from llm_client_v2 import generate_without_rag
        return generate_without_rag(question)
    except Exception:
        return "[纯LLM模式未启用]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tests = [
        "武汉力源信息技术股份有限公司本次发行股数是多少, 占发行后总股本的比例是多少?",
        "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目?",
        "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁, 持股比例和本公司关系是什么?",
        "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些?",
        "武汉兴图新科电子股份有限公司注册资本是多少?",
        "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
    ]

    for q in tests:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        result = answer_question(q)
        print(f"  表格查询: {result['is_table_query']}")
        print(f"  路由文档: {result['routed_docs']}")
        print(f"  响应时间: {result['response_time']}s")
        print(f"  RAG 答案: {result['rag_answer'][:150]}")
