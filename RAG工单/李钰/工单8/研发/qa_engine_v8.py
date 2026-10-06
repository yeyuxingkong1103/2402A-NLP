# -*- coding: utf-8 -*-
"""
问答引擎 V8 - Graph RAG 金融问答
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答

整合: 知识图谱 + Graph RAG + V6 混合检索 + LLM
"""
import os, sys, time, logging
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import knowledge_graph
import graph_retriever

logger = logging.getLogger(__name__)

_V6_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "工单6", "研发"))
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

_kg = None
_graph_retriever = None


def _ensure_ready():
    global _kg, _graph_retriever
    if _kg is None:
        _kg = knowledge_graph.KnowledgeGraph()
        kg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "financial_kg.json")
        if os.path.exists(kg_path):
            _kg.load_from_json(kg_path)
    if _graph_retriever is None:
        # 尝试加载 V6 文本检索
        text_retriever = None
        try:
            import qa_engine_v6
            v6_retriever = qa_engine_v6._get_retriever()
            text_retriever = v6_retriever
        except Exception as e:
            logger.warning(f"V6 文本检索不可用: {e}")
        _graph_retriever = graph_retriever.GraphRetriever(_kg, text_retriever)


def answer_question(question: str) -> Dict:
    """Graph RAG 问答"""
    start = time.time()
    _ensure_ready()

    # 1. Graph RAG 检索
    search_result = _graph_retriever.search(question)
    graph_result = search_result["graph"]
    fused_contexts = search_result["fused_contexts"]

    # 2. 生成答案
    rag_answer = _generate_with_graph(question, graph_result, fused_contexts)
    llm_answer = _generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "graph_result": {
            "center_nodes": graph_result["center_nodes"],
            "entity_linking": graph_result["entity_linking"],
            "graph_score": graph_result["graph_score"],
            "subgraph": graph_result["subgraph"],
            "graph_contexts": graph_result["graph_contexts"],
        },
        "fusion_score": search_result["fusion_score"],
        "rag_answer": rag_answer,
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= 3.0,
    }


def get_kg_visualization() -> Dict:
    """获取完整知识图谱可视化数据"""
    _ensure_ready()
    return _kg.to_visualization_data()


def query_graph_directly(query: str) -> Dict:
    """直接查询知识图谱 (问答+可视化)"""
    _ensure_ready()
    result = _graph_retriever.query_graph(query)
    return {
        "center_nodes": result["center_nodes"],
        "subgraph": result["subgraph"],
        "visualization": _kg.to_visualization_data(result["subgraph"]),
        "graph_contexts": result["graph_contexts"],
    }


def _generate_with_graph(question: str, graph_result: Dict,
                         contexts: list) -> str:
    """用图谱+上下文生成答案"""
    # 图谱直接命中优先
    graph_contexts = graph_result.get("graph_contexts", [])
    if graph_contexts:
        try:
            sys.path.insert(0, _V6_DIR)
            from llm_client_v2 import _call_openai_api
            system = (
                "你是金融知识图谱问答助手。优先根据知识图谱中的实体关系回答问题。\n"
                "知识图谱结构: 实体A --[关系]--> 实体B\n"
                "如果知识图谱中有相关信息, 直接引用。"
                "同时参考文本片段中的数据。"
            )
            context_text = "\n\n".join([f"【图谱】{c}" for c in graph_contexts[:10]] +
                                       [f"【文本】{c}" for c in contexts[len(graph_contexts):len(graph_contexts)+3]])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"知识图谱上下文:\n{context_text}\n\n问题: {question}"},
            ]
            answer = _call_openai_api(messages)
            if answer:
                return answer
        except Exception as e:
            logger.warning(f"LLM 调用不可用: {e}")

    # 降级: 图谱摘要
    if graph_contexts:
        return "[Graph RAG 降级回答]\n" + "\n".join(graph_contexts[:5])
    return "知识图谱中未找到相关信息"


def _generate_without_rag(question: str) -> str:
    try:
        sys.path.insert(0, _V6_DIR)
        from llm_client_v2 import generate_without_rag
        return generate_without_rag(question)
    except Exception:
        return "[纯LLM模式未启用]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "武汉力源信息技术股份有限公司的控股股东是谁,持股比例是多少?",
        "销售部有几个下属部门,大客户销售部有几个销售处?",
        "武汉兴图新科电子股份有限公司参与制定了什么标准?",
        "2008年中国IC市场增长最快和负增长的行业分别是哪些?",
    ]
    for q in tests:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        r = answer_question(q)
        print(f"  中心节点: {r['graph_result']['center_nodes']}")
        print(f"  图谱分数: {r['graph_result']['graph_score']}")
        print(f"  响应: {r['response_time']}s")
        print(f"  RAG: {r['rag_answer'][:120]}")
