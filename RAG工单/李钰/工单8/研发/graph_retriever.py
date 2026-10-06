# -*- coding: utf-8 -*-
"""
Graph RAG 检索器 - 图谱遍历 + 文本检索融合
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""
import os, sys, logging, json
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Graph RAG 检索器"""

    def __init__(self, kg, text_retriever=None):
        self.kg = kg
        self.text_retriever = text_retriever  # 可选: V6 混合检索

    def entity_linking(self, query: str) -> List[Dict]:
        """
        实体链接: 问题 → 图谱节点

        Returns:
            [{"node": str, "score": float, "matched": str}]
        """
        try:
            import jieba
            tokens = [t for t in jieba.cut(query) if len(t.strip()) > 1]
        except ImportError:
            tokens = list(query)

        matches = {}
        for token in tokens:
            nodes = self.kg.search_nodes(token, top_k=3)
            for nid in nodes:
                score = 1.0 if nid == token else 0.8 if token in nid else 0.5
                if nid not in matches or score > matches[nid]["score"]:
                    matches[nid] = {"node": nid, "score": score, "matched": token}

        results = sorted(matches.values(), key=lambda x: x["score"], reverse=True)
        logger.info(f"[实体链接] query='{query[:30]}' → {[r['node'] for r in results]}")
        return results

    def retrieve_subgraph(self, query: str, max_hops: int = 2,
                          top_k_nodes: int = 3) -> Dict:
        """
        子图检索

        Returns:
            {
                "center_nodes": [...],
                "subgraph": {"nodes": [...], "edges": [...]},
                "graph_score": float,
                "graph_contexts": [...],  # 图谱结构化描述, 供 LLM 使用
            }
        """
        # 1. 实体链接
        linked = self.entity_linking(query)
        center_nodes = [r["node"] for r in linked[:top_k_nodes]]

        if not center_nodes:
            return {"center_nodes": [], "subgraph": {"nodes": [], "edges": []},
                    "graph_score": 0.0, "graph_contexts": []}

        # 2. 子图遍历
        subgraph = self.kg.get_subgraph(center_nodes, max_hops)

        # 3. 计算图谱分数
        entity_hit_score = sum(r["score"] for r in linked[:top_k_nodes])
        subgraph_density = len(subgraph["edges"]) / max(len(subgraph["nodes"]), 1)
        graph_score = 0.7 * entity_hit_score + 0.3 * subgraph_density
        graph_score = min(graph_score, 1.0)

        # 4. 生成图谱上下文 (供 LLM)
        graph_contexts = self._build_graph_contexts(subgraph, center_nodes)

        return {
            "center_nodes": center_nodes,
            "subgraph": subgraph,
            "graph_score": round(graph_score, 4),
            "graph_contexts": graph_contexts,
            "entity_linking": linked,
        }

    def _build_graph_contexts(self, subgraph: Dict, center_nodes: List[str]) -> List[str]:
        """将子图转为 LLM 可读的上下文"""
        contexts = []

        # 中心节点描述
        for nid in center_nodes:
            neighbors = self.kg.get_neighbors(nid)
            parts = []
            for nb in neighbors[:10]:
                parts.append(f"[{nb['relation']}] {nb['node']}")
            if parts:
                contexts.append(f"实体 [{nid}]: " + "; ".join(parts))

        # 关系路径
        for edge in subgraph.get("edges", [])[:15]:
            contexts.append(f"关系: {edge['source']} --[{edge['relation']}]--> {edge['target']}")

        return contexts

    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Graph RAG 完整检索

        Returns:
            {
                "graph": {...},        # 图谱检索结果
                "text": [...],         # 文本检索结果 (可选)
                "fused_contexts": [...], # 融合后的上下文
                "fusion_score": float,
            }
        """
        # 1. 图谱检索
        graph_result = self.retrieve_subgraph(query)

        # 2. 文本检索 (如果可用)
        text_results = []
        if self.text_retriever:
            try:
                text_results = self.text_retriever.search(query, top_k=top_k)
            except Exception as e:
                logger.warning(f"文本检索不可用: {e}")

        # 3. 融合
        graph_contexts = graph_result.get("graph_contexts", [])
        text_contexts = [r.get("text", "") for r in text_results[:top_k] if isinstance(r, dict)]

        fused = graph_contexts + text_contexts

        graph_score = graph_result.get("graph_score", 0)
        text_score = max([r.get("score", 0) for r in text_results] + [0])
        fusion_score = 0.5 * graph_score + 0.3 * text_score

        return {
            "graph": graph_result,
            "text": text_results,
            "fused_contexts": fused,
            "fusion_score": round(fusion_score, 4),
        }

    def query_graph(self, query: str) -> Dict:
        """
        图谱问答 (直接从图谱取答案, 不依赖 LLM)

        示例:
          "武汉力源的控股股东是谁?" → 图谱直接返回 "武汉力源科技(35%)"
          "销售部有几个下属?" → 图谱返回 4 个
        """
        result = self.retrieve_subgraph(query, max_hops=2, top_k_nodes=5)
        return {
            "center_nodes": result["center_nodes"],
            "subgraph": result["subgraph"],
            "graph_contexts": result["graph_contexts"],
            "entity_linking": result["entity_linking"],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import knowledge_graph

    kg = knowledge_graph.KnowledgeGraph()
    kg.load_from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "financial_kg.json"))

    retriever = GraphRetriever(kg)

    queries = [
        "武汉力源信息技术股份有限公司的控股股东是谁?",
        "销售部有几个下属部门?",
        "武汉兴图新科参与制定了什么标准?",
    ]
    for q in queries:
        print(f"\n=== {q} ===")
        r = retriever.search(q)
        print(f"  中心节点: {r['graph']['center_nodes']}")
        print(f"  图谱分数: {r['graph']['graph_score']}")
        print(f"  融合分数: {r['fusion_score']}")
        for ctx in r["graph"]["graph_contexts"][:3]:
            print(f"  {ctx[:60]}...")
