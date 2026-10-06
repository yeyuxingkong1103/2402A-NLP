# -*- coding: utf-8 -*-
"""
Graph RAG 检索器 V2 - 路径检索 + 实体消歧 + 剪枝 + 重排
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import os, sys, logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class GraphRetrieverV2:
    """V2 Graph RAG 检索器"""

    def __init__(self, kg, text_retriever=None):
        self.kg = kg
        self.text_retriever = text_retriever

    def entity_linking(self, query: str) -> List[Dict]:
        """V2: 实体链接 + 消歧"""
        try:
            import jieba
            tokens = [t for t in jieba.cut(query) if len(t.strip()) > 1]
        except ImportError:
            tokens = list(query)

        # 多 token 搜索 + 合并
        matches = {}
        for token in tokens:
            nodes = self.kg.search_nodes(token, top_k=3)
            for n in nodes:
                nid = n["node"]
                score = n["score"]
                if nid in matches:
                    matches[nid]["score"] = max(matches[nid]["score"], score + 0.1)
                else:
                    matches[nid] = n

        results = sorted(matches.values(), key=lambda x: x["score"], reverse=True)

        # V9 新增: 实体消歧
        try:
            from entity_extractor_v2 import entity_disambiguation
            results = entity_disambiguation(query, results)
        except Exception:
            pass

        logger.info(f"[实体链接V2] '{query[:30]}' → {[r['node'] for r in results[:5]]}")
        return results

    def retrieve_with_paths(self, query: str, top_k_paths: int = 3) -> Dict:
        """V2: 路径检索 (而非简单 BFS)"""
        # 1. 实体链接
        linked = self.entity_linking(query)
        center_nodes = [r["node"] for r in linked[:3]]

        if len(center_nodes) < 2:
            # 单实体 → 退化到子图
            center = center_nodes[0] if center_nodes else None
            subgraph = self.kg.get_pruned_subgraph([center], max_hops=2) if center else {"nodes": [], "edges": []}
            return {
                "center_nodes": center_nodes, "subgraph": subgraph,
                "paths": [], "graph_score": 0.5,
                "graph_contexts": self._build_contexts(subgraph, center_nodes, []),
                "entity_linking": linked,
            }

        # 2. 路径查找 (两两中心节点)
        all_paths = []
        for i in range(len(center_nodes)):
            for j in range(i + 1, len(center_nodes)):
                paths = self.kg.find_paths(center_nodes[i], center_nodes[j], max_hops=3)
                all_paths.extend(paths)

        all_paths.sort(key=lambda x: x["score"], reverse=True)
        top_paths = all_paths[:top_k_paths]

        # 3. 路径驱动子图
        path_nodes = set()
        path_edges = []
        for p in top_paths:
            for n in p["path"]:
                path_nodes.add(n)
            for src, tgt, rel in zip(p["path"][:-1], p["path"][1:], p["relations"]):
                path_edges.append({"source": src, "target": tgt,
                                   "relation": rel, "weight": p["score"]})

        # 合并路径子图 + BFS 剪枝子图
        bfs_sg = self.kg.get_pruned_subgraph(center_nodes, max_hops=2)
        merged_nodes = path_nodes | set(bfs_sg["nodes"])
        merged_edges = path_edges + bfs_sg["edges"]
        # 去重
        seen = set()
        unique_edges = []
        for e in merged_edges:
            key = f"{e['source']}|{e['target']}|{e['relation']}"
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        subgraph = {"nodes": list(merged_nodes), "edges": unique_edges[:50]}

        # 4. 图谱分数
        entity_hit = sum(r["score"] for r in linked[:3])
        path_score = sum(p["score"] for p in top_paths) / max(len(top_paths), 1)
        node_imp = sum(self.kg.nodes.get(n, {}).get("importance", 0.5)
                       for n in center_nodes) / max(len(center_nodes), 1)
        graph_score = 0.4 * entity_hit + 0.4 * path_score + 0.2 * node_imp
        graph_score = min(graph_score, 1.0)

        return {
            "center_nodes": center_nodes,
            "subgraph": subgraph,
            "paths": top_paths,
            "graph_score": round(graph_score, 4),
            "graph_contexts": self._build_contexts(subgraph, center_nodes, top_paths),
            "entity_linking": linked,
        }

    def _build_contexts(self, subgraph: Dict, center_nodes: List[str],
                        paths: List[Dict]) -> List[str]:
        """V2: 结构化图谱上下文"""
        contexts = []

        # 1. 路径描述 (最精确)
        for p in paths[:5]:
            path_str = " → ".join([f"[{n}]" for n in p["path"]])
            rels_str = " → ".join(p["relations"])
            contexts.append(f"【路径】{path_str} | 关系: {rels_str} | 置信度: {p['score']}")

        # 2. 中心节点邻居
        for nid in center_nodes:
            neighbors = self.kg.get_neighbors(nid)
            if neighbors:
                parts = []
                for nb in neighbors[:8]:
                    arrow = "→" if nb["dir"] == "out" else "←"
                    parts.append(f"{arrow}[{nb['relation']}]{nb['node']}")
                contexts.append(f"【实体】{nid}: " + " ".join(parts))

        # 3. 关系边 (兜底)
        for edge in subgraph.get("edges", [])[:10]:
            contexts.append(
                f"【关系】{edge['source']} --[{edge['relation']}]--> {edge['target']}")

        return contexts

    def search(self, query: str, top_k: int = 5) -> Dict:
        """V2 融合检索"""
        # 图谱路径检索
        graph_result = self.retrieve_with_paths(query)

        # 文本检索
        text_results = []
        if self.text_retriever:
            try:
                text_results = self.text_retriever.search(query, top_k=top_k)
            except Exception as e:
                logger.warning(f"文本检索不可用: {e}")

        # 融合
        graph_contexts = graph_result["graph_contexts"]
        text_contexts = [r.get("text", "") for r in text_results[:top_k] if isinstance(r, dict)]

        # V9: 图谱上下文优先 (图谱精度更高)
        fused = graph_contexts + text_contexts

        graph_score = graph_result.get("graph_score", 0)
        text_score = max([r.get("score", 0) for r in text_results] + [0])
        fusion_score = 0.5 * graph_score + 0.3 * text_score + 0.2 * graph_score * text_score

        return {
            "graph": graph_result,
            "text": text_results,
            "fused_contexts": fused,
            "fusion_score": round(fusion_score, 4),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import knowledge_graph_v2 as kgv2

    kg = kgv2.KnowledgeGraphV2()
    kg.load_from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "financial_kg.json"))
    retriever = GraphRetrieverV2(kg)

    queries = [
        "武汉力源的控股股东是谁?",
        "销售部有几个下属?",
    ]
    for q in queries:
        r = retriever.search(q)
        print(f"\n=== {q} ===")
        print(f"  图谱分: {r['graph']['graph_score']}")
        for ctx in r['graph']['graph_contexts'][:3]:
            print(f"  {ctx[:70]}")
