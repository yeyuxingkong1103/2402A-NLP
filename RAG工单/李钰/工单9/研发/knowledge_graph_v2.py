# -*- coding: utf-8 -*-
"""
知识图谱 V2 - 路径分析 + 节点重要性 + 权重剪枝
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import json, os, logging, math
from typing import List, Dict, Set, Optional
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class KnowledgeGraphV2:
    """增强知识图谱"""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # id → {type, attributes, importance}
        self.out_edges: Dict[str, List[Dict]] = {}  # src → [{target, relation, weight}]
        self.in_edges: Dict[str, List[Dict]] = {}

    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_dict(data)

    def load_from_dict(self, data: Dict):
        # 节点
        for node in data.get("nodes", []):
            nid = node.get("id", "")
            if not nid: continue
            self.nodes[nid] = {
                "type": node.get("type", ""),
                "attributes": node.get("attributes", {}),
                "importance": 0.5,  # 默认重要性
            }
            self.out_edges.setdefault(nid, [])
            self.in_edges.setdefault(nid, [])

        # 边
        for edge in data.get("edges", []):
            src, tgt = edge.get("source", ""), edge.get("target", "")
            rel, w = edge.get("relation", ""), edge.get("weight", 0.8)
            if src and tgt:
                self.out_edges.setdefault(src, []).append(
                    {"target": tgt, "relation": rel, "weight": w})
                self.in_edges.setdefault(tgt, []).append(
                    {"source": src, "relation": rel, "weight": w})

        # 计算节点重要性 (入度 + 出度)
        for nid in self.nodes:
            in_deg = len(self.in_edges.get(nid, []))
            out_deg = len(self.out_edges.get(nid, []))
            total = in_deg + out_deg
            self.nodes[nid]["importance"] = min(1.0, 0.3 + 0.1 * total)

        logger.info(f"[KG V2] {len(self.nodes)} 节点, "
                     f"{sum(len(v) for v in self.out_edges.values())} 边")

    def search_nodes(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索节点 + 重要性评分"""
        query_lower = query.lower()
        results = []
        for nid, node in self.nodes.items():
            score = 0.0
            # 精确/模糊匹配
            if query_lower == nid.lower():
                score = 1.0
            elif query_lower in nid.lower() or nid.lower() in query_lower:
                score = 0.8 if nid.lower() in query_lower else 0.6
            # 重要性加成
            score += 0.2 * node.get("importance", 0.5)
            if score > 0:
                results.append({"node": nid, "score": round(min(score, 1.0), 4),
                                "type": node["type"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def find_paths(self, start: str, target: str, max_hops: int = 3) -> List[Dict]:
        """
        查找 start → target 的路径 (BFS)
        返回: [{"path": [...nodes...], "relations": [...], "score": float}]
        """
        if start not in self.nodes or target not in self.nodes:
            return []

        paths = []
        queue = deque([(start, [start], [], 0.0)])  # (node, path, relations, score)
        visited_paths = set()

        while queue and len(paths) < 10:
            node, path, rels, score = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue

            edges = self.out_edges.get(node, []) + self.in_edges.get(node, [])
            for edge in edges:
                tgt = edge.get("target", edge.get("source", ""))
                rel = edge.get("relation", "")
                if tgt == node:
                    tgt = edge.get("source", "")
                if tgt in path:
                    continue

                new_path = path + [tgt]
                new_rels = rels + [rel]
                # 路径分数: 边权重 × 节点重要性 / 路径长度 (衰减)
                edge_w = edge.get("weight", 0.5)
                node_imp = self.nodes.get(tgt, {}).get("importance", 0.5)
                new_score = (score + edge_w * node_imp) / (len(new_path) - 1)

                path_key = tuple(new_path)
                if path_key in visited_paths:
                    continue
                visited_paths.add(path_key)

                if tgt == target:
                    paths.append({
                        "path": new_path,
                        "relations": new_rels,
                        "score": round(new_score, 4),
                        "length": len(new_path) - 1,
                    })
                else:
                    queue.append((tgt, new_path, new_rels, new_score))

        paths.sort(key=lambda x: x["score"], reverse=True)
        return paths

    def get_pruned_subgraph(self, center_nodes: List[str],
                            max_hops: int = 2,
                            min_edge_weight: float = 0.3) -> Dict:
        """
        剪枝子图: 只保留权重 ≥ 阈值的边 + 重要性较高的节点
        """
        visited = set()
        all_edges = []
        frontier = set(center_nodes)

        for hop in range(max_hops):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)

                edges = self.out_edges.get(nid, []) + self.in_edges.get(nid, [])
                for edge in edges:
                    edge_w = edge.get("weight", 0.5)
                    if edge_w < min_edge_weight:
                        continue
                    tgt = edge.get("target", edge.get("source", ""))
                    if tgt == nid:
                        tgt = edge.get("source", "")
                    if tgt not in visited:
                        node_imp = self.nodes.get(tgt, {}).get("importance", 0.5)
                        # 剪枝: 低重要性节点 (除了中心节点的直接邻居)
                        if hop == 0 or node_imp >= 0.4:
                            all_edges.append({
                                "source": edge.get("source", nid),
                                "target": edge.get("target", tgt),
                                "relation": edge.get("relation", ""),
                                "weight": edge_w,
                            })
                            next_frontier.add(tgt)
            frontier = next_frontier

        # 去重边
        seen = set()
        unique_edges = []
        for e in all_edges:
            key = f"{e['source']}|{e['target']}|{e['relation']}"
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {
            "nodes": list(visited),
            "edges": unique_edges,
            "center_nodes": center_nodes,
        }

    def get_neighbors(self, nid: str) -> List[Dict]:
        neighbors = []
        for e in self.out_edges.get(nid, []):
            neighbors.append({"node": e["target"], "relation": e["relation"],
                              "weight": e["weight"], "dir": "out"})
        for e in self.in_edges.get(nid, []):
            neighbors.append({"node": e["source"], "relation": e["relation"],
                              "weight": e["weight"], "dir": "in"})
        return neighbors

    def to_visualization(self, subgraph: Dict = None) -> Dict:
        if subgraph:
            nodes = [{"id": n, "type": self.nodes.get(n, {}).get("type", ""),
                      "importance": self.nodes.get(n, {}).get("importance", 0.5)}
                     for n in subgraph["nodes"]]
            edges = subgraph["edges"]
        else:
            nodes = [{"id": nid, "type": nd["type"], "importance": nd["importance"]}
                     for nid, nd in self.nodes.items()]
            edges = []
            for src, lst in self.out_edges.items():
                for e in lst:
                    edges.append({"source": src, "target": e["target"],
                                  "relation": e["relation"], "weight": e["weight"]})
        return {"nodes": nodes, "edges": edges,
                "categories": [{"name": c} for c in ["Company", "Person", "Product",
                                                       "Standard", "Financial", "Industry"]]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = KnowledgeGraphV2()
    kg.load_from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "financial_kg.json"))

    # 路径查找
    paths = kg.find_paths("武汉力源", "武汉力源科技", max_hops=2)
    print(f"武汉力源 → 武汉力源科技: {len(paths)} 条路径")
    for p in paths[:3]:
        print(f"  {' → '.join(p['path'])} | score={p['score']}")

    # 剪枝子图
    sg = kg.get_pruned_subgraph(["武汉力源"], max_hops=2, min_edge_weight=0.5)
    print(f"剪枝子图: {len(sg['nodes'])} 节点, {len(sg['edges'])} 边")
