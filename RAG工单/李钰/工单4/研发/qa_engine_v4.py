# -*- coding: utf-8 -*-
"""
问答引擎 V4 - 图像内容解析 + 表格 + 文本
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

检索路由优先级: 图像 > 表格 > 文本
"""
import os
import sys
import time
import json
import logging
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v4 as config
import image_retriever
import image_understanding

logger = logging.getLogger(__name__)

# 引入 V3 (表格+文本) 模块
_V3_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "工单3", "研发"))
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)


_doc_images = None
_image_retriever = None


def _ensure_ready():
    global _doc_images, _image_retriever
    if _image_retriever is None:
        _image_retriever = image_retriever.ImageRetriever()
        _doc_images = image_understanding.load_preset_images()
        _image_retriever.load_images(_doc_images)


def _route_doc(question: str):
    """复用 V3 的文档路由"""
    try:
        import multi_pdf_manager
        mgr = multi_pdf_manager.get_manager()
        docs = mgr.route_query(question)
        if docs and len(docs) == 1:
            return docs[0].company
    except Exception as e:
        logger.warning(f"V3 路由不可用 ({e})")
    return None


def answer_question(question: str) -> Dict:
    """V4 完整 RAG 问答"""
    start = time.time()
    _ensure_ready()

    company_filter = _route_doc(question)
    is_image_q = image_retriever.is_image_query(question)

    rag_answer = ""
    answer_source = ""
    image_results = []
    table_results = []
    text_results = []

    # ===== 1. 图像检索 (优先) =====
    if is_image_q:
        image_results = _image_retriever.search(
            question, company_filter=company_filter, top_k=2)
        if image_results:
            rag_answer = _generate_from_images(question, image_results)
            answer_source = "image"

    # ===== 2. 表格检索 =====
    if not rag_answer:
        table_results, rag_answer = _call_v3_table(question, company_filter)
        if rag_answer:
            answer_source = "table"

    # ===== 3. 文本检索 =====
    if not rag_answer:
        text_results, rag_answer = _call_v3_text(question, company_filter)
        if rag_answer:
            answer_source = "text"

    # ===== 4. 纯 LLM 基线 =====
    llm_answer = _generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "is_image_query": is_image_q,
        "answer_source": answer_source,  # image / table / text
        "image_results": [{
            "image_type": r["image"].get("image_type"),
            "company": r["image"].get("company"),
            "score": r["score"],
            "hit_keywords": r["hit_keywords"],
        } for r in image_results],
        "table_results": table_results[:2] if isinstance(table_results, list) else [],
        "text_results": text_results[:3] if isinstance(text_results, list) else [],
        "rag_answer": rag_answer or "未检索到相关内容",
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= 3.0,
    }


def _generate_from_images(question: str, results: List[Dict]) -> str:
    """从图像描述生成答案"""
    # 优先用结构化数据
    for r in results:
        sd = r["image"].get("structured_data", {})
        if sd:
            structured = _query_structured(question, sd)
            if structured:
                return structured

    # 降级: 用描述文本
    context = _image_retriever.format_for_llm(results)
    answer = _call_llm_image(context, question)
    if answer:
        return answer

    # 最后降级: 拼接描述
    descs = [r["image"].get("description", "") for r in results[:2]]
    return "[图像描述] " + "；".join(descs)


def _query_structured(question: str, sd: Dict) -> str:
    """从结构化数据中直接提取答案"""
    # 规则匹配: 问题关键词 → 结构化数据键
    q_lower = question

    # 销售部部门构成
    if "销售部" in q_lower and ("几个" in q_lower or "部门构成" in q_lower or "下属" in q_lower):
        depts = sd.get("销售部下属部门", [])
        if depts:
            return f"销售部共有 {len(depts)} 个部门构成, 分别是: {', '.join(depts)}"

    # 大客户销售部销售处构成
    if "大客户销售部" in q_lower and ("销售处" in q_lower or "几个" in q_lower):
        places = sd.get("大客户销售部下属销售处", [])
        if places:
            return f"大客户销售部共有 {len(places)} 个销售处构成, 分别是: {', '.join(places)}"

    # 增长最快行业
    if "增长最快" in q_lower and "行业" in q_lower:
        fastest = sd.get("增长最快行业", [])
        if fastest:
            return f"增长率最快的行业是: {', '.join(fastest)}"

    # 负增长行业
    if "负增长" in q_lower and "行业" in q_lower:
        neg = sd.get("负增长行业", [])
        if neg:
            return f"负增长的行业是: {', '.join(neg)}"

    # 增长率数据
    if "增长率" in q_lower and "数据" in q_lower:
        data = sd.get("增长率数据", {})
        if data:
            lines = [f"{k}: {v}" for k, v in data.items()]
            return "增长率数据: " + "; ".join(lines)

    return ""


def _call_llm_image(context: str, question: str) -> str:
    """调用 LLM 处理图像描述"""
    system = (
        "你是金融文档图像问答助手。根据提供的图像描述和结构化数据回答问题。"
        "仅依据提供的内容, 不要编造。"
        "回答结构: 先给结论, 再引用数据。"
    )
    user = f"【图像描述】\n{context}\n\n【问题】\n{question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        sys.path.insert(0, _V3_DIR)
        from llm_client_v2 import _call_openai_api
        return _call_openai_api(messages) or ""
    except Exception:
        return ""


def _call_v3_table(question: str, company_filter: str):
    """复用 V3 表格检索"""
    try:
        import table_retriever as v3_table
        from multi_pdf_manager import get_manager
        # 加载预设表格
        import json
        presets_path = os.path.join(_V3_DIR, "预设表格数据.json")
        if os.path.exists(presets_path):
            with open(presets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("tables", [])
        else:
            tables = []
        retriever = v3_table.TableRetriever()
        retriever.load_tables(tables)
        results = retriever.search(question, company_filter=company_filter)
        if results:
            context = retriever.format_for_llm(results)
            answer = _call_llm_image(context, question) or retriever.format_for_llm(results)
            return results, answer
    except Exception as e:
        logger.warning(f"V3 表格检索不可用 ({e})")
    return [], ""


def _call_v3_text(question: str, company_filter: str):
    """复用 V2/V1 文本检索"""
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "工单2", "研发")))
        import vector_retrieval_v2 as v2_retrieval
        retriever = v2_retrieval.get_hybrid_retriever()
        results = retriever.search(question, top_k=5)
        if results:
            contexts = [r["text"] for r in results[:3]]
            context = "\n\n".join([f"【片段{i+1}】{c}" for i, c in enumerate(contexts)])
            answer = _call_llm_image(context, question) or context[:300]
            return results, answer
    except Exception as e:
        logger.warning(f"文本检索不可用 ({e})")
    return [], ""


def _generate_without_rag(question: str) -> str:
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "工单2", "研发")))
        from llm_client_v2 import generate_without_rag
        return generate_without_rag(question)
    except Exception:
        return "[纯LLM模式未启用]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?",
        "武汉力源信息技术股份有限公司招股意向书中,从2008年中国IC市场应用结构与增长图中可以看出,增长率最快的是哪个行业?负增长的是哪个行业?",
        "武汉力源信息技术股份有限公司本次发行股数是多少?",
        "武汉兴图新科电子股份有限公司注册资本是多少?",
    ]
    for q in tests:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        r = answer_question(q)
        print(f"  图像查询: {r['is_image_query']} | 来源: {r['answer_source']}")
        print(f"  响应: {r['response_time']}s")
        print(f"  RAG: {r['rag_answer'][:120]}")
